using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;
using Dojo;
using System.Collections.Generic;
using System.Linq;

namespace Examples.Bowling
{
    public class AIAgent : Agent
    {
        private GameBoard _board;
        private DojoConnection _connection;
        private int AgentID = 0;
        private float _feedbackReceived = 0;

        private bool _isDone = false;

        [Tooltip("Request decision every N seconds")]
        [SerializeField] private float _decisionRequestFrequency = 1f;
        [SerializeField] private bool _repeatActions = true;
        private AIAction _lastAction = AIAction.NO_MOVE;

        protected override void Awake()
        {
            base.Awake();
            _board = FindObjectOfType<GameBoard>();
            _board.OnEpisodeEnded += OnGameEnded;
            _board.OnFrameEnded += OnFrameEnded;
            
            _connection = FindObjectOfType<DojoConnection>();
            _connection.SubscribeRemoteMessages((long)NetOpCode.Feedback, OnRemoteFeedback);

#if UNITY_STANDALONE // && !UNITY_EDITOR
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];

                if (arg.Equals("-DecisionRequestFrequency") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var requestFreq))
                {
                    _decisionRequestFrequency = requestFreq;
                    ++idx;
                }
            }
#endif
            InvokeRepeating(nameof(DecisionRequestLoop), 0.0f, _decisionRequestFrequency);
        }

        private void FixedUpdate()
        {
            if (_repeatActions)
            {
                ExecuteAction(_lastAction);
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(_feedbackReceived);
            _feedbackReceived = 0;
            sensor.AddObservation(Time.realtimeSinceStartup);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            ActionSegment<int> actSegment = actions.DiscreteActions;
            var action = (AIAction)actSegment[0];
            ExecuteAction(action);
            _lastAction = action;
            
            if (_isDone)
            {
                EndEpisode();
                _isDone = false;
            }
        }

        private void ExecuteAction(AIAction action) {
            switch (action)
            {
                case AIAction.UP:
                    _board.HandleClientControl(NetCommand.Up);
                    break;

                case AIAction.DOWN:
                    _board.HandleClientControl(NetCommand.Down);
                    break;

                case AIAction.SHOOT:
                    _board.HandleClientControl(NetCommand.Shoot);
                    break;

                case AIAction.NO_MOVE:
                default:
                    break;
            }
        }

        private void DecisionRequestLoop()
        {
            RequestDecision();
        }

        private void OnFrameEnded(int frameCount, int score)
        {
            AddReward(score);
        }

        private void OnGameEnded()
        {
            _isDone = true;
        }

        private void OnRemoteFeedback(DojoMessage m)
        {
            var feedbackMessage = m.GetDecodedData<List<object>>();
            float feedback = Convert.ToSingle(feedbackMessage[0]);
            List<int> targets = (feedbackMessage[1] as IEnumerable<object>).Cast<object>().Cast<int>().ToList();
            if (targets.Contains(AgentID))
                _feedbackReceived += feedback;
        }
    }
}