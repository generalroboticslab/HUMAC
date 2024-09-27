using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;
using Unity.Netcode;
using Nakama.TinyJson;
using Dojo;
using Dojo.Netcode;
using Dojo.Recording;


namespace Examples.HideAndSeek
{
    public class PlayerController : NetworkBehaviour
    {
        private const string LOGSCOPE = "PlayerController";

        [SerializeField]
        private float _moveSpeed = 2f;

        [SerializeField, Tooltip("Rotation speed (in degrees)")]
        private float _rotateSpeed = 50f;

        [SerializeField]
        private InputActionAsset _playerActions;

        [SerializeField]
        private MaskedCamera _maskedCamera;

        [SerializeField]
        public AccumuCamera _accumuCamera;

        [SerializeField]
        private UIDocument inGameUI;

        private Label _timeoutTextUI;
        private Label _hiderCountUI;
        private Label _seekerCountUI;
        private Label _identityUI;

        private InputActionMap _playerControl;

        public Rigidbody _body;

        public SeekerHeruistic _policy;

        public Move _move;

        // public Hiderscript _hpolicy;
        private Vector3 _offset;
        private Vector3 _angleOffset;

        public Camera _globalCamera;
        private Camera _firstPersonCamera;

        private IPlayer _selfPlayer;
        public Camera CamEye => _firstPersonCamera;
        public Camera CamMasked => _maskedCamera.EnvCamera;
        public Camera CamAcc => _accumuCamera.FullCamera;

        public event Action OnControllerReady;

        private DojoConnection _connection;
        private DojoRecord _record;
        private DojoTransport _transport;

        private bool _enableFirstCamera = true;
        public bool EnableFirstCamera => _enableFirstCamera;
        private bool _enableMaskedCamera = true;
        public bool EnableMaskedCamera => _enableMaskedCamera;
        private bool _enableAccumuCamera = true;
        public bool EnableAccumuCamera => _enableAccumuCamera;

        public NetworkVariable<bool> undercontrol = new NetworkVariable<bool>(false);

        [HideInInspector] public NetworkVariable<int> AgentID = new NetworkVariable<int>(-1);

        public NetworkVariable<float> humanActionx = new NetworkVariable<float>(51f);
        public NetworkVariable<float> humanActionz = new NetworkVariable<float>(51f);
        private GameManager _gameManager;

        public NetworkVariable<float> agentspeed;

        public NetworkVariable<float> remainingDistance;

        public float change_angle;
        public List<float> change_angle_list = new List<float>();

        public float hider_left;

        private Unity.Netcode.Components.NetworkTransform networkTransform;

        public NetworkVariable<bool> clear_cam_flag = new NetworkVariable<bool>(false);


        private void Awake()
        {
            _playerControl = _playerActions.actionMaps[0];

            _body = GetComponentInChildren<Rigidbody>();

            _policy = GetComponentInChildren<SeekerHeruistic>();

            _move = GetComponentInChildren<Move>();
            _offset = Vector3.zero;
            _angleOffset = Vector3.zero;

            _globalCamera = Camera.main;
            _firstPersonCamera = GetComponentInChildren<Camera>();
            _selfPlayer = GetComponent<IPlayer>();

            var uiRoot = inGameUI.rootVisualElement;
            inGameUI.rootVisualElement.style.display = DisplayStyle.None;
            _timeoutTextUI = uiRoot.Q<Label>("Timeout");
            _hiderCountUI = uiRoot.Q<Label>("HiderCount");
            _seekerCountUI = uiRoot.Q<Label>("SeekerCount");
            _identityUI = uiRoot.Q<Label>("Identity");

            _connection = FindObjectOfType<DojoConnection>();
            _record = FindObjectOfType<DojoRecord>();

            _gameManager = GameManager.Instance;
            networkTransform = GetComponent<Unity.Netcode.Components.NetworkTransform>();

            // CamAcc.enabled = true;


#if UNITY_STANDALONE
            var args = Environment.GetCommandLineArgs();
            //Debug.Log(args);
            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];
                if (arg.Equals("-DisableFirstCamera"))
                {
                    _enableFirstCamera = false;
                }
                else if (arg.Equals("-DisableMaskedCamera"))
                {
                    _enableMaskedCamera = false;
                }
                else if (arg.Equals("-DisableAccumuCamera"))
                {
                    _enableAccumuCamera = false;
                }
            }
#endif
        }

        private void Update()
        {


            if (IsOwner)
            {
               
                HandleHumanInput();
                _timeoutTextUI.text = TimeSpan.FromSeconds(GameManager.Instance.GetMatchTimeout()).ToString(@"mm\:ss");
                _hiderCountUI.text = $"Hiders: {GameManager.Instance.HiderCount}";
                _seekerCountUI.text = $"Seekers: {GameManager.Instance.SeekerCount}";
            }
            UpdateHumanAction();
            
            Collider[] hitColliders = Physics.OverlapBox(new Vector3(0f,0f,0f),new Vector3(50f,50f,50f));
            foreach (Collider col in hitColliders)
            {
                if (col.CompareTag("Hider"))
                {
                    GameObject hider = col.gameObject;
                    change_angle_list = hider.transform.GetComponentInChildren<Hiderscript>().change_angle_list;

                }
            }

        }

        private void FixedUpdate()
        {
            if (IsServer)
            {
                _body.MovePosition(transform.position + Time.deltaTime * _moveSpeed * _offset);
                _offset = Vector3.zero;

                _body.MoveRotation(_body.rotation * Quaternion.Euler(_angleOffset * Time.fixedDeltaTime));
                _angleOffset = Vector3.zero;
            }  

            hider_left = 0f;
            Collider[] hitColliders = Physics.OverlapBox(new Vector3(0f,0f,0f),new Vector3(50f,5f,50f));
            foreach (Collider col in hitColliders)
            {
                if (col.CompareTag("Hider"))
                {
                    hider_left = hider_left + 1f;
                }
            }  

            if (clear_cam_flag.Value)
            {
                _accumuCamera.ClearAccumulation();
                // _accCamSens.ClearAccumulation();
                if (IsServer)
                {
                    clear_cam_flag.Value = false;
                }else
                {
                    ChangeFlagServerRpc();
                }
            }
        }

        [ServerRpc(RequireOwnership = false)]
        private void ChangeFlagServerRpc()
        {
            clear_cam_flag.Value = false;
        }


        public override void OnDestroy()
        {
            try
            {
                _globalCamera.enabled = true;
            }
            catch { }
        }

        private void OnCollisionEnter(Collision other)
        {
            var player = other.gameObject.GetComponent<IPlayer>();
            var playerController = other.gameObject.GetComponent<PlayerController>();
            if (IsServer && player != null)
            {
                if (_transport.GetUserByNetcodeID(OwnerClientId, out var user1) &&
                    _connection.MatchClients.TryGetValue(user1, out var role1) &&
                    _transport.GetUserByNetcodeID(playerController.OwnerClientId, out var user2) &&
                    _connection.MatchClients.TryGetValue(user2, out var role2))
                {
                    var eventData = new List<string>() { user1.UserId, role1.ToString(), user2.UserId, role2.ToString() };
                    _record.DispatchEvent(RecordEvent.PlayerCollision, JsonWriter.ToJson(eventData));
                }

                if (_selfPlayer.IsHider && !player.IsHider)
                {
                    Debug.Log("Hider collide");

                    if (AgentID.Value >= 0)
                    {
                        GameManager.Instance.HiderHasDied(AgentID.Value);
                    }
                    else if (GameManager.Instance.HostType == GameHostType.WaitingRoom)
                    {
                        GameManager.Instance.HiderHasDied(OwnerClientId);
                    }
                    else
                    {
                        HiderCollidedClientRpc();
                    }
                }
                else if (!_selfPlayer.IsHider && player.IsHider)
                {
                    Debug.Log("Seeker collide");

                    var controller = player.GetComponent<PlayerController>();
                    GameManager.Instance.SeekerHasCaught(AgentID.Value, controller.AgentID.Value);
                }
            }
        }

        public override void OnNetworkSpawn()
        {
            if (IsOwner && IsClient)
            {
                OnGainedOwnership();
            }
            OnControllerReady?.Invoke();

            _transport = NetworkManager.Singleton.NetworkConfig.NetworkTransport as DojoTransport;

            //Control if the camera is on when running the game
            _accumuCamera.IsEnabled= true;
            _accumuCamera.FollowGlobalCamera(_globalCamera);
            //_maskedCamera.IsEnabled = true;
            //_maskedCamera.FollowGlobalCamera(_globalCamera);
            _globalCamera.enabled = true;
        }

        public override void OnNetworkDespawn()
        {
            if (IsOwner && IsClient)
            {
                OnLostOwnership();
            }
        }

        public override void OnGainedOwnership()
        {
            if (IsClient)
            {
                Debug.Log($"{LOGSCOPE}: Gained Ownership");
                _globalCamera.enabled = false;

                _firstPersonCamera.enabled = _enableFirstCamera;
                _playerControl.Enable();
                _accumuCamera.IsEnabled = _enableAccumuCamera;
                _accumuCamera.FollowGlobalCamera(_globalCamera);
                _maskedCamera.IsEnabled = _enableMaskedCamera;
                _maskedCamera.FollowGlobalCamera(_globalCamera);
                inGameUI.rootVisualElement.style.display = DisplayStyle.Flex;
                _identityUI.text = "You're " + (GameManager.Instance.IsHider ? "Hider" : "Seeker");
                SwitchCamera(1);
                _record.DispatchEvent(RecordEvent.PlayerStateChange, $"Spawned {_rotateSpeed} {_moveSpeed} {DojoRecordEncode.Encode(transform)}");
            }
        }

        public override void OnLostOwnership()
        {
            if (IsClient)
            {
                Debug.Log($"{LOGSCOPE}: Lost Ownership");
                _globalCamera.enabled = true;
                _firstPersonCamera.enabled = false;
                _playerControl.Disable();
            }
        }

        public void SwitchCamera(int cameraIdx)
        {
            if (cameraIdx == 1 && !_enableFirstCamera)
            {
                return;
            }
            else if (cameraIdx == 2 && !_enableMaskedCamera)
            {
                return;
            }
            else if (cameraIdx == 3 && !_enableAccumuCamera)
            {
                return;
            }

            TurnOffCamera(_firstPersonCamera);
            TurnOffCamera(_maskedCamera.EnvCamera);
            TurnOffCamera(_accumuCamera.FullCamera);

            switch (cameraIdx)
            {
                case 2:
                    {
                        TurnOnCamera(_maskedCamera.EnvCamera);
                        break;
                    }

                case 3:
                    {
                        TurnOnCamera(_accumuCamera.FullCamera);
                        break;
                    }

                case 1:
                default:
                    {
                        TurnOnCamera(_firstPersonCamera);
                        break;
                    }
            }
        }

        private void TurnOnCamera(Camera cam)
        {
            cam.depth = 1.0f;
            //cam.clearFlags = CameraClearFlags.Skybox;
            //cam.cullingMask = LayerMask.GetMask(new[] { "Everything", "Default", "TransparentFX", "Ignore Raycast", "PostProcessing", "Water", "UI", "Players" });
        }

        private void TurnOffCamera(Camera cam)
        {
            cam.depth = -100.0f;
            //cam.clearFlags = CameraClearFlags.Color;
            //cam.cullingMask = LayerMask.GetMask(new[] { "Nothing" });
        }

        private void HandleHumanInput()
        {
            
            if (_playerControl["PositionClick"].IsPressed())
            {
                Screen.SetResolution(Screen.width,Screen.height,Screen.fullScreen);
                Vector3 p3 = Mouse.current.position.ReadValue();
                MoveToCLickedPosition(p3,true);
            }
        }

        [ServerRpc(RequireOwnership = false)]
        private void UpdateHumanActionServerRpc()
        {
            if (_playerControl["PositionClick"].IsPressed())
            {

                Screen.SetResolution(Screen.width,Screen.height,Screen.fullScreen);
                Vector3 p3 = Mouse.current.position.ReadValue();
                Ray ray = _globalCamera.ScreenPointToRay(p3);
                RaycastHit hit;
                if (Physics.Raycast(ray, out hit))
                {
                    p3 = hit.point;
                }
                Debug.Log("Huamn Input:"+p3);
                humanActionx = new NetworkVariable<float>(p3.x);
                humanActionz = new NetworkVariable<float>(p3.z);
            }
        }


         private void UpdateHumanAction()
        {
            if (_connection.IsClient)
            {
                UpdateHumanActionServerRpc();
            }
        }


        public void MoveToCLickedPosition(Vector3 p3,bool IshumanControl)
        {
            if (IsServer)
            {
                if (IshumanControl)
                {
                    Ray ray = _globalCamera.ScreenPointToRay(p3);
                    RaycastHit hit;
                    if (Physics.Raycast(ray, out hit))
                    {
                        transform.GetComponent<Navigation>().SetDestination(hit.point);
                    }
                }
                else
                {
                    // Debug.Log("Screen name:"+UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
                    transform.GetComponent<Navigation>().SetDestination(p3);
                }
               
            }
            else
            {
                Debug.Log("M1111");
                ClickServerRpc(p3,IshumanControl);
            }
        }

        [ServerRpc(RequireOwnership = false)]
        private void ClickServerRpc(Vector3 p3, bool IshumanControl)
        {
            Debug.Log("M2222");
            if (IshumanControl)
            {
                Ray ray = _globalCamera.ScreenPointToRay(p3);
                RaycastHit hit;
                if (Physics.Raycast(ray, out hit))
                {
                    transform.GetComponent<Navigation>().SetDestination(hit.point);
                }
            }
            else
            {
                Debug.Log("M3333");
                if (!transform.GetComponent<Navigation>().enabled)
                {
                    transform.GetComponent<Navigation>().enabled = true;
                }
                transform.GetComponent<Navigation>().SetDestination(p3);
            }
        }

        public void CheckReachDestination()
        {
            if (!IsServer)
            {
                CheckReachDestinationServerRpc();
            }
        }

        [ServerRpc(RequireOwnership = false)]
        private void CheckReachDestinationServerRpc()
        {
            if (transform.GetComponent<Navigation>().enabled)
            {
                if (transform.GetComponent<Navigation>().reachedDestination())
                {
                    transform.GetComponent<Navigation>().Stop();
                    transform.GetComponent<Navigation>().enabled = false;
                    _policy.enabled = true;
                }
            }
        }
        


        public void SetRotationSpeed(float speed)
        {
            if (IsServer)
            {
                _rotateSpeed = speed;
            }
            else
            {
                SetRotationSpeedServerRpc(speed);
                _record.DispatchEvent(RecordEvent.PlayerStateChange, $"RotationSpeed {speed}");
            }
        }

        public void SetMoveSpeed(float speed)
        {
            if (IsServer)
            {
                _moveSpeed = speed;
            }
            else
            {
                SetMoveSpeedServerRpc(speed);
                _record.DispatchEvent(RecordEvent.PlayerStateChange, $"MoveSpeed {speed}");
            }
        }

        private void RecordAction(string action)
        {
            _record.DispatchEvent(RecordEvent.PlayerAction, $"{action} {DojoRecordEncode.Encode(transform)}");
        }


        [ServerRpc]
        private void SetRotationSpeedServerRpc(float speed)
        {
            SetRotationSpeed(speed);
        }

        [ServerRpc]
        private void SetMoveSpeedServerRpc(float speed)
        {
            SetMoveSpeed(speed);
        }

        [ClientRpc]
        private void HiderCollidedClientRpc()
        {
            if (IsOwner && GameManager.Instance.HostType == GameHostType.JoinLeave)
            {
                GameManager.Instance.HiderHasDied();
            }
        }

        public void StopAgent()
        {
            if (!IsServer)
            {
                StopAgentServerRpc();
            }
        }

        [ServerRpc(RequireOwnership = false)]
        public void StopAgentServerRpc()
        {
            // Debug.Log("Agent Stopped");
            _policy.enabled = false;  
        }

        // public void ResumeAgent()
        // {
        //     if (!IsServer)
        //     {
        //         ResumeAgentServerRpc();
        //     }else
        //     {
        //         if (transform.GetComponentInChildren<Navigation>().enabled)
        //         {
        //             if (transform.GetComponentInChildren<Navigation>().remainingDistance <= 0.1f)
        //             {
        //                 transform.GetComponentInChildren<Navigation>().ResetPath();
        //                 transform.GetComponentInChildren<Navigation>().enabled = false;
        //                 _policy.enabled = true;
        //             }
        //         }
        //     }
        // }

        // [ServerRpc(RequireOwnership = false)]
        // public void ResumeAgentServerRpc()
        // {
        //     if (transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().enabled)
        //     {
        //         if (transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().remainingDistance <= 0.1f)
        //         {
        //             transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().ResetPath();
        //             transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().enabled = false;
        //             _policy.enabled = true;
        //         }
        //     }
        // }

        public void moveAgent(Vector3 move)
        {
            if (!IsServer)
            {
                moveAgentServerRpc(move);
            }
        }

        [ServerRpc(RequireOwnership = false)]
        public void moveAgentServerRpc(Vector3 move)
        {
            _policy.enabled = false;  
            transform.GetComponentInChildren<Navigation>().enabled = true;
            transform.GetComponentInChildren<Rigidbody>().velocity = Vector3.zero;
            transform.GetComponentInChildren<Rigidbody>().angularVelocity = Vector3.zero;
            
            transform.forward = move;
            transform.position += move*4f*Time.deltaTime; 
        }

        public float Getspeed()
        {
            if (!IsServer)
            {
                GetspeedServerRpc();
            }
            else
            {
                agentspeed.Value = transform.GetComponentInChildren<Rigidbody>().velocity.magnitude;
            }
            return agentspeed.Value;
        }

        [ServerRpc(RequireOwnership = false)]
        public void GetspeedServerRpc()
        {
            agentspeed.Value =  transform.GetComponentInChildren<Rigidbody>().velocity.magnitude;
        }


        // public void StartAgent()
        // {
        //     if (!IsServer)
        //     {
        //         StartAgentServerRpc();
        //     }else
        //     {
        //         transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().ResetPath();
        //         transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().enabled = false;
        //         // _policy.enabled = true;  
        //         _move.enabled = true;
        //     }
        // }

        // [ServerRpc(RequireOwnership = false)]
        // public void StartAgentServerRpc()
        // {
        //     transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().ResetPath();
        //     transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().enabled = false;
        //     // _policy.enabled = true;  
        //     _move.enabled = true;
            
        // }

        public void StartAgentH()
        {
            if (!IsServer)
            {
                StartAgentHServerRpc();
            }else
            {
                transform.GetComponent<Navigation>().Stop();
                transform.GetComponent<Navigation>().enabled = false;
                _policy.enabled = true;  
            }
        }

        [ServerRpc(RequireOwnership = false)]
        public void StartAgentHServerRpc()
        {
            Debug.Log("Start Agent");
            transform.GetComponent<Navigation>().Stop();
            transform.GetComponent<Navigation>().enabled = false;
            _policy.enabled = true;  
            // _move.enabled = true;
            
        }

        public void StopAgentH()
        {
            if (!IsServer)
            {
                StopAgentHServerRpc();
            }else
            {
                // transform.GetComponentInChildren<Navigation>().ResetPath();
                transform.GetComponent<Navigation>().enabled = true;
                _policy.enabled = false;
                //completely stop the agent from moving
                   
                // _move.enabled = true;
            }
        }

        [ServerRpc(RequireOwnership = false)]
        public void StopAgentHServerRpc()
        {
            // transform.GetComponentInChildren<UnityEngine.AI.NavMeshAgent>().ResetPath();
            transform.GetComponent<Navigation>().enabled = true;
            _policy.enabled = false;  
            // _move.enabled = true;
            
        }


        public void OnImitationLearning()
        {
            if (!IsServer)
            {
                OnImitationLearningServerRpc();
            }
        }

        [ServerRpc(RequireOwnership = false)]
        public void OnImitationLearningServerRpc()
        {
            undercontrol.Value = true;
            Debug.Log("Switched to true");
        }

        public void OffImitationLearning()
        {
            if (!IsServer)
            {
                OffImitationLearningServerRpc();
            }
        }

        [ServerRpc(RequireOwnership = false)]
        public void OffImitationLearningServerRpc()
        {
            undercontrol.Value = false;
        }

        public void Teleport(Vector3 position)
        {
            if (IsServer)
            {
                // navmeshagent.ResetPath();
                networkTransform.Teleport(position, transform.rotation, transform.localScale);
            }
        }


    }
}