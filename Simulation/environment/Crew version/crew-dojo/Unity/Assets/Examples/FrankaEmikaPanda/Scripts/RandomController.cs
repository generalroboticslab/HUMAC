using UnityEngine;
using Mujoco;
using Dojo;

namespace Examples.FrankaEmikaPanda
{
    public class RandomController : MonoBehaviour
    {
        [SerializeField]
        private float _updateFrequency = 0.5f;

        private MjActuator[] _actuators;

        // randomly control the actuators
        private void Awake()
        {
            _actuators = FindObjectsOfType<MjActuator>();
            Debug.Assert(_actuators.Length > 0);
        }

        private void Start()
        {
            var conn = FindObjectOfType<DojoConnection>();
            if (!conn.IsClient)
            {
                RandomMove();
            }
        }

        private void RandomMove()
        {
            foreach (var actuator in _actuators)
            {
                var range = actuator.CommonParams.CtrlRange;
                actuator.Control = Mathf.Clamp(actuator.Control + Random.Range(range.x, range.y), range.x, range.y);
            }

            Invoke(nameof(RandomMove), _updateFrequency);
        }
    }
}