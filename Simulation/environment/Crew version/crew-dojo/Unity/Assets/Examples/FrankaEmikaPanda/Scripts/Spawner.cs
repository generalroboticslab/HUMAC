using UnityEngine;
using Unity.Netcode;
using Dojo.Netcode;

namespace Examples.FrankaEmikaPanda
{
    public class Spawner : MonoBehaviour
    {
        [SerializeField]
        private DojoNetcodeObjectPool _objectPool;

        [SerializeField]
        private GameObject _robotArmPrefab;

        private void Start()
        {
            NetworkManager.Singleton.OnServerStarted += OnServerStarted;
        }

        private void OnServerStarted()
        {
            if (NetworkManager.Singleton.IsServer)
            {
                SpawnRobotArm();
            }
        }

        public void SpawnRobotArm()
        {
            _ = _objectPool.GetNetworkObject(_robotArmPrefab).gameObject;
        }
    }
}