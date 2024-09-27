using UnityEngine;
using Dojo;
using Dojo.UI;
using Dojo.UI.Feedback;

namespace Examples.FrankaEmikaPanda
{
    public class HumanInterface : FeedbackInterface
    {
        private const string LOGSCOPE = "HumanInterface";

        [SerializeField]
        private DojoMenu _menu;

        //[SerializeField]
        //private InputActionAsset _feedbackActions;

        private DojoConnection _connection;
        //private InputActionMap _feedbackControl;

        protected override void Awake()
        {
            base.Awake();
            _connection = FindObjectOfType<DojoConnection>();

            // register callbacks
            _connection.OnJoinedMatch += ToggleUI;
            _connection.OnLeftMatch += ToggleUI;
            _connection.OnRoleChanged += m => ToggleUI();

            OnPositiveButton += OnButtonPositive;
            OnNeutralButton += OnButtonNeutral;
            OnNegativeButton += OnButtonNegative;

            //_feedbackControl = _feedbackActions.actionMaps[0];
            //_feedbackControl.Enable();

            Visible = false;
        }

        //private void Update()
        //{
        //    if (Visible)
        //    {
        //        if (_feedbackControl["Positive"].WasPressedThisFrame())
        //        {
        //            OnButtonPositive();
        //        }
        //        if (_feedbackControl["Neutral"].WasPressedThisFrame())
        //        {
        //            OnButtonNeutral();
        //        }
        //        if (_feedbackControl["Negative"].WasPressedThisFrame())
        //        {
        //            OnButtonNegative();
        //        }
        //    }
        //}

        private void ToggleUI()
        {
            Visible = _connection.HasJoinedMatch && _connection.Role == DojoNetworkRole.Viewer;
        }

        #region Button Callbacks

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

        private void SendFeedback(int val)
        {
            //var message = val.ToString();
            //var targets = _menu.SelectedFeedbackClients;
            //if (targets.Count > 0)
            //{
            //    _connection.SendStateMessage((long)NetOpCode.Feedback, message, targets: targets);
            //}
            //else
            //{
            //    Debug.LogWarning($"{LOGSCOPE}: Button clicked but no feedback target client selected");
            //}
        }

        #endregion Button Callbacks
    }
}