using UnityEngine;

namespace Examples.FrankaEmikaPanda
{
    [DefaultExecutionOrder(-1)]
    public class GameManager : MonoBehaviour
    {
        [SerializeField]
        private float _mujocoTimestep = 0.0005f;

        private void Awake()
        {
            Time.fixedDeltaTime = _mujocoTimestep;
        }
    }
}