using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Dojo;
using System.Collections.Generic;
using System.Collections;
using System.Linq;

namespace Examples.HideAndSeek
{
    public class AIAgent : Agent
    {
        [Header("Configs")]
        [SerializeField] private float _agentMoveSpeed = 5.0f;

        [SerializeField] private float _agentRotationSpeed = 60.0f;

        [Tooltip("Request decision every N seconds")]
        [SerializeField] private float _decisionRequestFrequency = 3.0f;

        [SerializeField] private bool _repeatActions = true;

        [HideInInspector] public int AgentID = -1;
        public PlayerController _playerController;

        public bool IsPlayerAlive => _playerController != null;

        private Vector3 _lastAction=new Vector3(20f,0f,20f);

        private GameManager _gameManager;
        private DojoConnection _connection;

        private float _feedbackReceived = 0;

        public int counter;
        private Transform bd;
        bool isrepeating;

        public bool _isActive = true;


        private bool _imitationLearning = false;

        // bool already_reset = false;

        int last_change_angle_list_len;
        float change_angle;

        Vector3 last_decision_position;
        Vector3 last_decision;

        protected override void Awake()
        {
            base.Awake();
            isrepeating = false;
            // collectrepeat = false;
#if UNITY_STANDALONE // && !UNITY_EDITOR
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                
                var arg = args[idx];

                if (arg.Equals("-MoveSpeed") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var moveSpeed))
                {
                    _agentMoveSpeed = moveSpeed;
                    ++idx;
                }

                if (arg.Equals("-RotationSpeed") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var rotSpeed))
                {
                    _agentRotationSpeed = rotSpeed;
                    ++idx;
                }

                if (arg.Equals("-DecisionRequestFrequency") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var requestFreq))
                {
                    _decisionRequestFrequency = requestFreq;
                    ++idx;
                }
            }
#endif
            var sensors = GetComponents<CameraSensorComponent>();
            foreach (var sensor in sensors)
            {
                sensor.Camera = Camera.main;
            }
            _gameManager = GameManager.Instance;
            _connection = FindObjectOfType<DojoConnection>();
            _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);
            _connection.SubscribeRemoteMessages((long)NetOpCode.ImitationLearning, OnImitationLearning);
            _connection.SubscribeRemoteMessages((long)NetOpCode.CancelIL, OffImitationLearning);
        }

 private void FixedUpdate()
        {   
        }

        private void DecisionRequestLoop()
        {
            // Debug.Log("Decision Request Loop: "+_isActive);
            if (_isActive)
            {
                // bool gamePaused = _gameManager.GamePaused;
                if (IsPlayerAlive)
                {
                    // if (_gameManager.GameRunning)
                    // {
                    counter = counter + 1;
                    RequestDecision();
                    change_angle = 0f;
                    //}
                }
            }
        }

        public void SubscribeController(PlayerController controller)
        {
            _playerController = controller;
            _playerController.AgentID.Value = AgentID;
            _playerController.SetMoveSpeed(_agentMoveSpeed);
            _playerController.SetRotationSpeed(_agentRotationSpeed);

            // Debug.Log("ACC Cam:"+ _playerController.CamAcc.enabled+" "+_playerController.CamAcc.depth);
            // Debug.Log("Global Cam:"+ _playerController._globalCamera.enabled +" "+_playerController._globalCamera.depth);
            
            // _playerController.CamAcc.depth = -10f;
            // _playerController._globalCamera.depth = 100f;
            // _playerController._globalCamera.enabled = true;
            // _playerController.CamAcc.enabled = true;

            bd = controller.transform;


            var sensors = GetComponents<CameraSensorComponent>();
            foreach (var sensor in sensors)
            {
                if (sensor.SensorName.Contains("FirstPerson"))
                {
                    sensor.Camera = _playerController.CamEye;
                    sensor.enabled = _playerController.EnableAccumuCamera;
                }
                else if (sensor.SensorName.Contains("Masked"))
                {
                    sensor.Camera = _playerController.CamAcc;
                    sensor.enabled = _playerController.EnableMaskedCamera;
                }
                else if (sensor.SensorName.Contains("Accumulative"))
                {
                    sensor.Camera = _playerController.CamAcc;
                    sensor.enabled = _playerController.EnableAccumuCamera;
                }
            }
            
            if (!isrepeating)
            {

                isrepeating = true;
                InvokeRepeating(nameof(DecisionRequestLoop), 0.0f, _decisionRequestFrequency);
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            if (bd != null)
            {
                sensor.AddObservation(bd.position.x);
                sensor.AddObservation(bd.position.z);
                sensor.AddObservation(_gameManager.num_ep.Value);
                sensor.AddObservation(_imitationLearning);
                sensor.AddObservation(_gameManager.Num_hider_alive);
            }
            else
            {
                sensor.AddObservation(101f);
                sensor.AddObservation(102f);
                sensor.AddObservation(0);
                sensor.AddObservation(0);
                sensor.AddObservation(0);
            }
           
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Debug.Log("OnActionReceived:"+actions.ContinuousActions[0]+" "+actions.ContinuousActions[1]);
            ActionSegment<float> actSegment = actions.ContinuousActions;
            Vector3 p3 = new Vector3(actSegment[0],0f,actSegment[1]);
            // Vector3 p3 = new Vector3(51f,0f,51f);
            
            // Debug.Log(p3);
            if(p3.x == 51f & p3.z == 51f)
            {   
                
                if (_playerController.GetComponentInChildren<Navigation>().enabled)
                {
                    _playerController.GetComponentInChildren<Navigation>().Stop();
                    _playerController.StartAgentH();
                }
                Debug.Log("Heuristic:"+_playerController._policy.enabled);
            }
            else
            {
                // comment or uncomment this one to decide between collecting data and imitation learning
     
                if (_playerController._policy.enabled)
                {
                    _playerController.StopAgentH();
                }
                
                //make sure this is the p3 is 2.1f from curent position
                if (Vector3.Distance(_playerController.transform.position,p3) < 1.5f)
                {
                    p3 = _playerController.transform.position + 1.5f*(p3-_playerController.transform.position).normalized;
                }

                ExecuteAction(p3);
                last_decision = p3;
                last_decision_position = _playerController.transform.position;
                last_decision_position.y = 0f;

            }
        }

        private void ExecuteAction(Vector3 p3)
        {
            // bool gamePaused = _gameManager.GamePaused;
            // if (!GameManager.Instance.GameRunning || !IsPlayerAlive || gamePaused)
            //     return;

            _playerController.MoveToCLickedPosition(p3,false);
            _lastAction = p3;

        }


        private void OnRemoteFeedback(DojoMessage m)
        {
            var feedbackMessage = m.GetDecodedData<List<object>>();
            float feedback = Convert.ToSingle(feedbackMessage[0]);
            List<int> targets = (feedbackMessage[1] as IEnumerable<object>).Cast<object>().Cast<int>().ToList();
            if (targets.Contains(AgentID))
                _feedbackReceived += feedback;
        }

        private void OnImitationLearning(DojoMessage m)
        {
            if (!_connection.IsServer)
                return;
            var imitationLearningMessage = m.GetDecodedData<List<object>>();
            int target = (int)imitationLearningMessage[0];
            Debug.Log("target:"+target+","+"AgentID: "+AgentID);
            if (target == AgentID)
            {
                _imitationLearning = true;
                Debug.Log("IL started");
                Debug.Log("IL here: "+_imitationLearning);
            }
        }

        private void OffImitationLearning(DojoMessage m)
        {
            if (!_connection.IsServer)
                return;
            var OffimitationLearningMessage = m.GetDecodedData<List<object>>();
            int target = (int)OffimitationLearningMessage[0];
            if (target == AgentID)
            {
                Debug.Log("IL stopped");
                _imitationLearning = false;

            }
        }

        public void HiderCaught(bool ishider)
        {
            if (ishider)
            {
                AddReward(-1.0f);
            }
            else
            {
                AddReward(1.0f);
            }
            // _isActive = false;
            EndEpisode();
            //_trajectoryID += 1;
        }

        public void StepsReached()
        {
            // _isActive = false;
            EndEpisode();
            //_trajectoryID += 1;
        }

        public void StartRequestingDecisions()
        {

            if (!_connection.IsServer)
                return;
            _isActive = true;
        }

        public IEnumerator WaitAndStartRequestingDecisions()
        {
            // Debug.Log("Waiting for decision called");
            yield return null; // waits one frame
            if (!_connection.IsServer)
                yield return null;
            _isActive = true;
        }




    }
}