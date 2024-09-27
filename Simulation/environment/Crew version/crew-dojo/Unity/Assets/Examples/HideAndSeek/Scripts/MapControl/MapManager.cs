using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Unity.Netcode;
using Dojo.Netcode;
using System.Text;
using System.IO;

namespace Examples.HideAndSeek
{
    [RequireComponent(typeof(DojoNetcodeObjectPool))]
    public class MapManager : MonoBehaviour
    {
        private const string LOGSCOPE = "MapManager";

        private DojoNetcodeObjectPool _pool;

        [SerializeField]
        private Transform _ground;

        public List<List<int>> MapObstacles { get; private set; } = new();
        // public List<Vector2Int> SeekerSpawnPoints { get; private set; } = new();

        public List<Vector2Int> AgentSpawnPoints { get; private set; } = new();
        private readonly List<Tuple<NetworkObject, GameObject>> _obstacles = new();

        public event Action OnMapReady;

        // 0 for empty space
        // x for spawn point for either hider or seeker
        // 1,2,3,4 are 4 possible obstacles

        public int NumRows { get; private set; } = 0;
        public int NumCols { get; private set; } = 0;
        public float NumRowsHalf { get; private set; } = 0;
        public float NumColsHalf { get; private set; } = 0;
        public Vector3 GroundScale { get; private set; } = Vector3.zero;
        public List<Vector3> agent_positions = new();

        private void Awake()
        {
            _pool = GetComponent<DojoNetcodeObjectPool>();

            var scale = _ground.localScale;
            _ground.localScale.Set(scale.x, 1.0f, scale.z);
            
        }

    

        public string rand_map(string map)
        {
            //randomize map
            UnityEngine.Random.InitState(GameManager.Instance.random_seed);
            Debug.Log("Random Seed for Map:"+GameManager.Instance.random_seed);

            StringBuilder sb = new StringBuilder(map);
            for(int i = 1 ; i <= 3;i=i+2)
            {
                for (int j = 1 ; j <= 3;j= j+2)
                {   
                    int obNumber = UnityEngine.Random.Range(49, 53); 
                    int posrow = UnityEngine.Random.Range(4,8);
                    int poscol = UnityEngine.Random.Range(4,8);
                    sb[101*((posrow+10*j)-1)+2*(poscol+10*i)-1] = (char)obNumber;
                }

            }

            
            int obNumber1 =  UnityEngine.Random.Range(49, 53); 
            int posrow1 =  UnityEngine.Random.Range(4,7);
            int poscol1 =  UnityEngine.Random.Range(4,7);
            sb[101*((posrow1+10*2)-1)+2*(poscol1+10*2)-1] = (char)obNumber1;
            
            //agent spwan spot
            for(int i = 7; i <= 10; i++)
            {
                for (int j = 7; j <= 43; j++)
                {
                    sb[101*(i-1)+2*(j)-1] = 'x';
                }
            }
            
            for(int i = 40; i <= 43; i++)
            {
                for (int j = 7; j <= 43; j++)
                {
                    sb[101*(i-1)+2*(j)-1] = 'x';
                }
            }

            for(int i = 11; i <= 39; i++)
            {
                for (int j = 7; j <= 10; j++)
                {
                    sb[101*(i-1)+2*(j)-1] = 'x';
                }
            }

            for(int i = 11; i <= 39; i++)
            {
                for (int j = 7; j <= 10; j++)
                {
                    sb[101*(i-1)+2*(j)-1] = 'x';
                }
            }


            for(int i = 11; i <= 39; i++)
            {
                for (int j = 40; j <= 43; j++)
                {
                    sb[101*(i-1)+2*(j)-1] = 'x';
                }
            }



            
            return sb.ToString();


        }

        public bool LoadMap(string map)
        {
            // split by rows
            var lines = map.Split(Environment.NewLine).ToList();

            // filter out empty lines or comments
            lines = lines.Select(line => line.Trim())
                .Where(line => !string.IsNullOrEmpty(line) && !line.StartsWith("//")).ToList();
            if (lines.Count == 0)
            {
                return false;
            }

            // validate and convert map
            var numRows = lines.Count;
            var numCols = -1;
            var tmpMap = new List<List<int>>();
            foreach (var line in lines)
            {
                var mapRow = line.Split(",")
                    .Where(val => !string.IsNullOrEmpty(val) && (val == "x" || val == "y" || int.TryParse(val, out _)))
                    .Select(val => val == "x" ? -1 : (val == "y" ? -2 : int.Parse(val)))
                    .ToList();
                if (numCols < 0)
                {
                    numCols = mapRow.Count;
                }
                if (mapRow.Count == 0 || numCols != mapRow.Count)
                {
                    return false;
                }
                tmpMap.Add(mapRow);
            }
            MapObstacles = tmpMap;

            // init map info
            NumRows = MapObstacles.Count;
            NumCols = MapObstacles[0].Count;
            NumRowsHalf = NumRows * 0.5f;
            NumColsHalf = NumCols * 0.5f;
            GroundScale = _ground.localScale * 50.0f;
            GroundScale = new(
                GroundScale.x/ NumCols,
                GroundScale.y,
                GroundScale.z/ NumRows
            );

            // scan spawn points
            agent_positions.Clear();
            AgentSpawnPoints.Clear();

            Debug.Log(MapObstacles.ToString());

            for (var rowId = 0; rowId < numRows; ++rowId)
            {
                for (var colId = 0; colId < numCols; ++colId)
                {
                    if (MapObstacles[rowId][colId] == -1)
                    {
                        AgentSpawnPoints.Add(new(rowId, colId));
                    }
    
                    
                }
            }
            Debug.Assert(AgentSpawnPoints.Count > 0, $"{LOGSCOPE}: Should have at least 1 spawn point!");

            // allocate obstacles
            //ClearObstacles();
            AllocateObstacles();

            // invoke ready
            OnMapReady?.Invoke();

            return true;
        }

        public void ClearObstacles()
        {
            _obstacles.ForEach(obj => _pool.ReturnNetworkObject(obj.Item1, obj.Item2));
            //_obstacles.Clear();
        }

        private void AllocateObstacles()
        {
            UnityEngine.Random.InitState(GameManager.Instance.random_seed);
            if (MapObstacles.Count == 0 || MapObstacles[0].Count == 0)
            {
                return;
            }

            
            for (var rowId = 0; rowId < NumRows; ++rowId)
            {
                for (var colId = 0; colId < NumCols; ++colId)
                {
                    var objIdx = MapObstacles[rowId][colId];
                    if (objIdx > 0)
                    {
                        var prefab = _pool.GetPrefabAt(objIdx - 1);
                        var netObj = _pool.GetNetworkObject(prefab);
                        _obstacles.Add(Tuple.Create(netObj, prefab));

                        var obj = netObj.gameObject;
                        var pos = obj.transform.localPosition;
                        obj.transform.localPosition = new Vector3(
                            (colId - NumColsHalf + 0.5f) * GroundScale.x,
                            pos.y,
                            (NumRowsHalf - rowId - 0.5f) * GroundScale.z

                        );
                        
                        obj.transform.rotation = Quaternion.Euler(0f, UnityEngine.Random.Range(0f, 360f), 0f);

                        var scale = obj.transform.localScale;
                        //obj.transform.localScale = new Vector3(10f*GroundScale.x, scale.y, GroundScale.z);
                    }
                }
            }
        }

    }
}