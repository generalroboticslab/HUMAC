using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using System;
using Nakama.TinyJson;
using Dojo;
using Dojo.UI;
using Dojo.UI.Feedback;
using Dojo.Recording;
using System.Linq;

namespace Examples.HideAndSeek
{
    public class HumanInterface : FeedbackInterface
    {
        private const string LOGSCOPE = "HumanInterface";

        [SerializeField]
        private DojoMenu _menu;

        [SerializeField]
        private InputActionAsset _feedbackActions;

        private DojoConnection _connection;
        private InputActionMap _feedbackControl;

        private DojoRecord _record;

        private InputAction _leftstick;

        private InputAction _rightrigger;

        private InputAction _leftMouseClick;
        private InputAction _rightMouseClick;

        private PlayerController _controller;
        private Camera _mainCamera;
        public bool _isControllingAgent = false;

        [SerializeField]
        private InputActionAsset playerActions;
        public GameManager _gameManager;
        int last_ep_number = 0 ; 
        Vector3 boxcenter;
        bool vieweing;
        Vector3 currentDestination = new Vector3(51f,51f,51f);

        Queue<int> AgentIDQueue = new Queue<int>();

        Vector2 move;

        protected override void Awake()
        {
            base.Awake();
            

            _gameManager = GameManager.Instance;

            _connection = FindObjectOfType<DojoConnection>();
            _record = FindObjectOfType<DojoRecord>();

            // register callbacks
            _connection.OnJoinedMatch += ToggleUI;
            _connection.OnLeftMatch += ToggleUI;
            _connection.OnRoleChanged += m => ToggleUI();

            OnPositiveButton += OnButtonPositive;
            OnNeutralButton += OnButtonNeutral;
            OnNegativeButton += OnButtonNegative;

            OnTakeControlButton += OnButtonTakeControl;

            // _leftstick = new InputAction(binding: "<Gamepad>/leftStick");
            // _leftstick.performed += ctx => move = ctx.ReadValue<Vector2>();
            // _leftstick.canceled += ctx => move = Vector2.zero;
            // _leftstick.Enable();

            // _rightrigger = new InputAction(binding: "<Gamepad>/RightShoulder");
            // _rightrigger.performed += ctx =>  OnRightTrigger();
            // _rightrigger.Enable();

            _leftMouseClick = new InputAction(binding: "<Mouse>/leftButton");
            _leftMouseClick.performed += ctx => OnLeftMouseClick();
            _leftMouseClick.Enable();

            _rightMouseClick = new InputAction(binding: "<Mouse>/rightButton");
            _rightMouseClick.performed += ctx => OnRightMouseClick();
            _rightMouseClick.Enable();            



            _feedbackControl = _feedbackActions.actionMaps[0];
            if (_elements.Contains(Elements.DISCRETE))
            {
                _feedbackControl.Enable();
            }



            _mainCamera = Camera.main;

            boxcenter.x = _mainCamera.transform.position.x;
            boxcenter.y = 0f;
            boxcenter.z = _mainCamera.transform.position.z;
            // Debug.Log("Box Center:"+boxcenter);

            Visible = false;

        }

        private void Update()
        {
            if (Visible)
            {
                if (_feedbackControl["Positive"].WasPressedThisFrame())
                {
                    OnButtonPositive();
                }
                if (_feedbackControl["Neutral"].WasPressedThisFrame())
                {
                    OnButtonNeutral();
                }
                if (_feedbackControl["Negative"].WasPressedThisFrame())
                {
                    OnButtonNegative();
                }
            }

            if (_controller == null)
            {
                // Debug.Log("AutomaticallyTakeControl");
                // _mainCamera.depth = 100f;
                AutomaticallyTakeControl();
            }
            else
            {
                if (currentDestination != new Vector3(51f,51f,51f) && Vector3.Magnitude(_controller.transform.position - currentDestination) < 0.76f)
                {
                    var eventData = new List<object>() { _controller.AgentID.Value};
                    _connection.SendStateMessage((long)NetOpCode.CancelIL, JsonWriter.ToJson(eventData));
                }
                CheckReachDestination();
            }
        }

        private void CheckReachDestination()
        {
            _controller.CheckReachDestination();
        }

        private void AutomaticallyTakeControl()
        {
            if (AgentIDQueue.Count == 0)
            {
                Collider[] hitColliders = Physics.OverlapBox(boxcenter,new Vector3(50f,50f,50f));
                foreach (Collider col in hitColliders)
                {
                    if (col.CompareTag("Seeker"))
                    {
                        AgentIDQueue.Enqueue(col.transform.GetComponentInChildren<PlayerController>().AgentID.Value);
                    }
                }
            }

            // Debug.Log("AgentIDQueue:"+ String.Join(",", AgentIDQueue));

            if (_connection.IsViewer && _controller == null)
            {
                Collider[] hitColliders = Physics.OverlapBox(boxcenter,new Vector3(50f,50f,50f));
                foreach (Collider col in hitColliders)
                {
                    if (col.CompareTag("Seeker"))
                    {
                        var controller = col.transform.GetComponentInChildren<PlayerController>();
                        if (controller != null && !controller.undercontrol.Value && controller.AgentID.Value == AgentIDQueue.Peek())
                        {
                            // controller.OnImitationLearning();
                            _controller = controller;
                            _mainCamera.depth = -10f;
                            _controller.CamAcc.depth = 10f;

                            Debug.Log("Selected Agent:"+_controller.AgentID.Value);
                            
                            // var eventData = new List<object>() { _controller.AgentID.Value};
                            // _connection.SendStateMessage((long)NetOpCode.ImitationLearning, JsonWriter.ToJson(eventData));

                            _controller.transform.GetComponent<MeshRenderer>().material.color = new Color(1.0f, 0.647f, 0.0f);

                            // _controller.StopAgent();
                            break;
                        }
                    }
                }
            }
        }


        private void ToggleUI()
        {
            Visible = _connection.HasJoinedMatch && _connection.Role == DojoNetworkRole.Viewer;
        }

        #region Button Callbacks

        private void OnButtonTakeControl()
        {
            var targets = _menu.SelectedFeedbackAIPlayers;

            var targetAgentIDs = targets.ConvertAll(target => int.Parse(target.Split(char.Parse("-")).ToList().Last()));
            
            if (targets.Count != 1)
            {
                Debug.LogWarning($"{LOGSCOPE}: Button clicked but only 1 target client can be selected");
                return;
            }
            var targetAgentID = targetAgentIDs[0];

            //fix me: should plus the number of hider
            // var eventData = new List<object>() { targetAgentID+1 };
            // _connection.SendStateMessage((long)NetOpCode.ImitationLearning, JsonWriter.ToJson(eventData));
            // _isControllingAgent = !_isControllingAgent;
            // _takeControl.SetMode(_isControllingAgent ? TakeControl.Mode.ReleaseControl : TakeControl.Mode.TakeControl);
        }

        private void OnButtonPositive()
        {
            SendFeedback(1);
        }

        private void OnButtonNegative()
        {
            SendFeedback(-1);
        }

        private void OnButtonNeutral()
        {
            SendFeedback(0);
        }

        private void SendFeedback(float val)
        {
            var targets = _menu.SelectedFeedbackAIPlayers;
            var targetAgentIDs = targets.ConvertAll(target => int.Parse(target.Split(char.Parse("-")).ToList().Last()));

            var eventData = new List<object>() { val, targetAgentIDs };

            if (targets.Count > 0)
            {
                _connection.SendStateMessage((long)NetOpCode.Feedback, JsonWriter.ToJson(eventData));
            }
            else
            {
                Debug.LogWarning($"{LOGSCOPE}: Feedback provided but no feedback target client selected");
            }
        }

        #endregion Button Callbacks

        #region Mouse Click Callbacks

        // private void handlejoystick(Vector2 move) // control the seeker
        // {


        //     if (_connection.IsViewer && _controller != null)
        //     {
                
        //         if (move.x == 0f && move.y == 0f)
        //         {
        //             _controller.StartAgentH();
        //         }else
        //         {
        //             var eventData = new List<object>() { _controller.AgentID.Value};
        //             _connection.SendStateMessage((long)NetOpCode.ImitationLearning, JsonWriter.ToJson(eventData));
                    
        //             _controller.StopAgent();

        //             move.Normalize();
        //             Vector3 Move = new Vector3(move.x,0f,move.y);
        //             _controller.moveAgent(Move);
        //         }
        //     }
        // }

        private void OnLeftMouseClick() // control the seeker
        {
            if (_connection.IsViewer && _controller != null)
            {
                if (Input.GetMouseButtonDown(0)) // 0 is for left mouse button, 1 for right mouse button, 2 for middle mouse button
                {
                    Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                    RaycastHit hit;

                    if (Physics.Raycast(ray, out hit))
                    {
                        // Get the point where the ray hit
                        Vector3 clickPoint = hit.point;
                        var eventData = new List<object>() { _controller.AgentID.Value};
                        // Debug.Log("message:"+JsonWriter.ToJson(eventData));
                        _connection.SendStateMessage((long)NetOpCode.ImitationLearning, JsonWriter.ToJson(eventData));
                        
                        _controller.StopAgent();
                        clickPoint.y = 0f; 

                        currentDestination = clickPoint;

                        _controller.transform.GetComponent<Navigation>().enabled = true;
                        _controller.MoveToCLickedPosition(clickPoint,false);
                    }
                }

                
            }
        }


        private void OnRightMouseClick() // switch seeker
        {
            if (_connection.IsViewer && _controller != null )
            {
                currentDestination = new Vector3(51f,51f,51f);
                // _controller.CamAcc.enabled = false;
                // _mainCamera.enabled = true;
                int a = AgentIDQueue.Dequeue();
                AgentIDQueue.Enqueue(a);

                _mainCamera.depth = 10f;
                _controller.CamAcc.depth = -10f;

                Debug.Log("UnSelected Agent:"+_controller.AgentID.Value);
                var eventData = new List<object>() { _controller.AgentID.Value};
                _connection.SendStateMessage((long)NetOpCode.CancelIL, JsonWriter.ToJson(eventData));

                _isControllingAgent = !_isControllingAgent;
                _takeControl.SetMode(_isControllingAgent ? TakeControl.Mode.ReleaseControl : TakeControl.Mode.TakeControl);
                
                //start the policy again
                _controller.transform.GetComponent<MeshRenderer>().material.color = Color.red;
                _controller.StartAgentH();
                // _controller.OffImitationLearning();

                _controller = null;
                
            }
        }

        #endregion Mouse Click Callbacks
    }
}