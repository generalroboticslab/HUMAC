using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

namespace Examples.HideAndSeek
{
public class Hiderscript : MonoBehaviour
{

    [SerializeField]
    public int speed;

    [SerializeField]
    public int hider_detect_range = 10;

    [SerializeField]
    public int obstacledetectrange = 5;

    Vector3 Direction;

    public List<GameObject> target_list = new List<GameObject>();

    public List<GameObject> last_target_list = new List<GameObject>();

    public List<GameObject> wall_list = new List<GameObject>();

    public List<float> change_angle_list = new List<float>();

    public bool is_run_in_middle;

    int time_counter;

    public bool special_case;

    public int keep_action_counter = 0;

    List<int> result_list;

    Vector3 last_direction;

    int last_seeker_num;

    void Start()
    {
    

        time_counter = 0;
        
        //initialize the run in the middle to be false
        is_run_in_middle = false;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        special_case = false;
        time_counter = time_counter + 1;   

        target_list.Clear();
        RaycastHit[] hits1 = Physics.SphereCastAll(new Ray(transform.position, Vector3.up),hider_detect_range); //figure why 5 doesn't work
        foreach (var hit in hits1)
        {
            if (hit.collider.CompareTag("Seeker") && !target_list.Contains(hit.collider.gameObject))
            {
                target_list.Add(hit.collider.gameObject);
            }
        }

        // Debug.Log("Last:"+string.Join(",",last_target_list));
        // Debug.Log("This:"+string.Join(",",target_list));
        //detect wall
        wall_list.Clear();

        RaycastHit[] hits = Physics.SphereCastAll(new Ray(transform.position, Vector3.up),5f); 
        foreach (var hit in hits)
        {
            if (hit.collider.CompareTag("Wall") && !wall_list.Contains(hit.collider.gameObject))
            {
                wall_list.Add(hit.collider.gameObject);
            }
        }

        Direction = calculate_run_away_direction(transform,target_list,wall_list);

            
        Direction.Normalize();

        Direction = Direction*4f;


        for (int j = 0;j <= 36;j++)
        {
            
            Vector3 ray_d11 = rotate(Direction,Mathf.PI/180*10f*j,true); // change this angle
            Vector3 ray_d12 = rotate(Direction,Mathf.PI/180*10f*(j+1),true); // change this angle
            Vector3 ray_d13 = rotate(Direction,Mathf.PI/180*10f*(j-1),true); // change this angle


            // destination = transform.position + ray_d11;
            // // Direction = ray_d11;
            if (NoObstacle(transform,ray_d11,obstacledetectrange)  &&  NoObstacle(transform,ray_d12,obstacledetectrange) && NoObstacle(transform,ray_d13,obstacledetectrange)  )
            {
                // Debug.Log("turn cw"+j);
                Direction = ray_d11;
                break;
            }

            
            // if (NoObstacle(transform,ray_d21,obstacledetectrange) &&  NoObstacle(transform,Direction,obstacledetectrange) && NoObstacle(transform,ray_d22,obstacledetectrange) )
            // {
            //     Debug.Log("turn not cw"+j);
            //     Direction = ray_d21;
            //     break;
            // }

        }

        for (int j = 0;j <= 1800;j++) // fix me 
        {
            
            Vector3 ray_d11 = rotate(Direction,Mathf.PI/180*0.1f*j,true); // change this angle
            Vector3 ray_d21 = rotate(Direction,Mathf.PI/180*0.1f*j,false);

            // destination = transform.position + ray_d11;
            // Direction = ray_d11;
            if (NoWall(transform,ray_d11,obstacledetectrange))
            {
                // Debug.Log("turn cw"+j);
                Direction = ray_d11;
                break;
            }

            
            if (NoWall(transform,ray_d21,obstacledetectrange))
            {
                // Debug.Log("turn not cw"+j);
                Direction = ray_d21;
                break;
            }

        }


        //handle special cases
        //1. get away from the wall when no seeker is chasing
        if (target_list.Count == 0 && wall_list.Count != 0)
        {
            Direction = get_away_from_wall(wall_list,Direction);
            Direction.Normalize();

            // Direction = Direction*1f;
            // //think about how to do this
            // Vector3 destination1 = transform.position + Direction;
            // destination1.y = 0f;
        
            // _agent.destination = destination1;    
        }

    //2. run from the middle of two seekers
        if (target_list.Count >1 && wall_list.Count != 0)
        {
            if (!is_run_in_middle)
            {
                
                Direction = run_in_middle(wall_list,target_list,Direction);
                is_run_in_middle = true;
                special_case = false;
                
                keep_action_counter = 0;
                // Vector3 destination1 = transform.position + Direction;
                // destination1.y = 0f;
            
                // _agent.destination = destination1;
                // Debug.Log("Start Run in the middle");
            }
            else
            {
                if (new_seeker_in_range(last_target_list,target_list))
                {
                    Direction = run_in_middle(wall_list,target_list,Direction);
                    keep_action_counter = 0;
                }
            }

        }
            
        else
        {

            if(is_run_in_middle && keep_action_counter < 30)
            {
                if (target_list.Count ==2)
                {
                    Direction = transform.forward; // this line is important
                }
                keep_action_counter++;
                // Debug.Log("keep run in the middle");
            }
            else
            {
                is_run_in_middle = false;
                keep_action_counter = 0;
            }
        }

        Direction.Normalize();

        if (Vector3.Dot(transform.forward,Direction) < 0f && special_case)
        {
            // Debug.Log("Special");
            if (Mathf.Abs(transform.position.x) >= 20f )
            {
                if (Mathf.Abs(Mathf.Abs(transform.position.z) - 20f) < 3f)
                {
                    Direction = transform.forward;
                }
            }else
            {
                if (Mathf.Abs(Mathf.Abs(transform.position.x) - 20f) < 3f)
                {
                    Direction = transform.forward;
                }
            }
            // Direction = transform.forward;
        }   

        // Debug.Log("Out of function D2:"+Direction);

        last_target_list.Clear();
        RaycastHit[] hits2 = Physics.SphereCastAll(new Ray(transform.position, Vector3.up),hider_detect_range); //figure why 5 doesn't work
        foreach (var hit in hits2)
        {
            if (hit.collider.CompareTag("Seeker") && !last_target_list.Contains(hit.collider.gameObject))
            {
                last_target_list.Add(hit.collider.gameObject);
            }
        }


        transform.forward = Direction;
        transform.position = transform.position +Direction*speed*Time.deltaTime; 


    }

    Vector3 run_in_middle(List<GameObject> wall_list, List<GameObject> target_list, Vector3 Direction)
    {
        if (wall_list.Count ==0)
        {
            List<int> new_result_list = check_seeker_all_on_one_side(target_list,transform);
            result_list = new_result_list;
            //Debug.Log("New result_list "+string.Join(",",new_result_list));
            if (result_list[0] == 0)
            {
                List<GameObject> target_list_copy = new List<GameObject>();
                target_list_copy.Add(target_list[result_list[1]]);
                target_list_copy.Add(target_list[result_list[2]]);
                target_list = target_list_copy; 

                var target1 = target_list[0];
                var target2 = target_list[1];

                var A= transform.position - target1.transform.position ;
                var B = transform.position - target2.transform.position ;

                //Debug.Log("Run from the middle of two seekers, No wall");
                float coef= Vector3.Magnitude(B)/(Vector3.Magnitude(B)+Vector3.Magnitude(A));
                Direction = -A*(coef)-B*(1-coef);
                Direction.Normalize();
            }

        }
        else if (wall_list.Count == 1) //1 wall with multiple seekers
        {
            //calcuate if all the seekers are on one side of the wall

            List<int> new_result_list = check_seeker_all_on_one_side_wall(target_list,wall_list[0],transform);


            if (new_result_list[0] == 0) //case in a wall where seekers are on the different side run from the other side
            {
                // Debug.Log("New result_list "+string.Join(",",new_result_list));
                if (new_result_list[2] != -1)
                {
                    var target1 = target_list[new_result_list[1]];
                    var target2 = target_list[new_result_list[2]];
                    
                    var A= transform.position - target1.transform.position ;
                    var B = transform.position - target2.transform.position ;

                    A.y = 0f;
                    B.y = 0f;

                    //change this to calculated angle
                    //Debug.Log("Run from the middle of two seekers, wall");
                    float coef= Vector3.Magnitude(B)/(Vector3.Magnitude(B)+Vector3.Magnitude(A));
                    Direction = -A*(coef)-B*(1-coef);
                    Direction.Normalize();
                    
                }
                else //the biggest gap is between a seeker and a wall 
                {
                    var target1 = target_list[new_result_list[1]];
  
                    var A= transform.position - target1.transform.position ;

                    A.y = 0f;

                    
                    if (Vector3.Dot(A,wall_list[0].transform.forward) >= 0)
                    {
                        Direction = - wall_list[0].transform.forward;
                    }else
                    {
                        Direction = wall_list[0].transform.forward;
                    }
                    Direction.Normalize();

                }


            }
            else //case in a wall where seekers are on the same side run from the other side
            {
                var target1 = target_list[0];
                var A= transform.position - target1.transform.position ;
                A.y = 0f;

                //Debug.Log(time_counter*Time.deltaTime+"s: One Wall two Seeker");

                Vector3 tempDirection;
                if (Vector3.Dot(wall_list[0].transform.forward,A) >= 0)
                {
                    tempDirection = wall_list[0].transform.forward;
                }else
                {
                    tempDirection = -wall_list[0].transform.forward;
                }
                tempDirection.Normalize();
                tempDirection = tempDirection*1f;
                Direction = tempDirection;
            }
        }
        else //corner with multiple seekers
        {
            special_case = false;


            List<int> new_result_list = check_seeker_all_on_one_side_corner(target_list,wall_list,transform);

            //Debug.Log("Result List: "+string.Join(",",new_result_list));
            
            if (new_result_list[0] == 0)
            {

                if (new_result_list[2] >= 0) 
                {

                    //Debug.Log("Run from the middle of seekers in corner");
                    var target1 = target_list[new_result_list[1]];
                    var target2 = target_list[new_result_list[2]];

                    var A= transform.position - target1.transform.position ;
                    var B = transform.position - target2.transform.position ;

                    A.y = 0f;
                    B.y = 0f;
                    //change this to calculated angle
                    // Debug.Log("Here");
                    float coef= Vector3.Magnitude(B)/(Vector3.Magnitude(B)+Vector3.Magnitude(A));
                    Direction = -A*(coef)-B*(1-coef);
                    Direction.Normalize();
                    
                }
                else
                {
                    //Debug.Log(time_counter*Time.deltaTime+"s: Corner two Seeker from diffrent direction, run from the side");
                    
                    var target1 = target_list[new_result_list[1]];
                    var wall = wall_list[-new_result_list[2]-1];

                    Vector3 tempDirection;
                    Vector3 norm;

                    tempDirection = wall.transform.forward;
                    norm = rotate(wall.transform.forward,Mathf.PI/2,true);


                
                    if (!IsPositionWalkable(transform.position + tempDirection*4f))
                    {
                        tempDirection = -tempDirection;
                    }

                    tempDirection = tempDirection + norm;

                    tempDirection.Normalize();
                

                    Direction = tempDirection;

                    
                }
            
            }
            else //all seekers are on one side of the corner, run from the other side
            {
                var target1 = target_list[0];
                var target2 = target_list[1];

                //Debug.Log(time_counter*Time.deltaTime+"s: Corner two Seeker from same direction, run from the side");
                Vector3 tempDirection;
                Vector3 norm;
                float d2 = Mathf.Max(Mathf.Abs(Vector3.Dot(target1.transform.position,rotate(wall_list[0].transform.forward,Mathf.PI/2,true))),Mathf.Abs(Vector3.Dot(target1.transform.position,rotate(wall_list[1].transform.forward,Mathf.PI/2,true))));
                if (d2 == Mathf.Abs(Vector3.Dot(target1.transform.position,rotate(wall_list[0].transform.forward,Mathf.PI/2,true))))
                {
                    tempDirection = wall_list[1].transform.forward;
                    norm = rotate(wall_list[1].transform.forward,Mathf.PI/2,true);
                    
                }else
                {
                    tempDirection = wall_list[0].transform.forward;
                    norm = rotate(wall_list[0].transform.forward,Mathf.PI/2,true);
                }


                if (!IsPositionWalkable(transform.position + tempDirection*4f))
                {
                    tempDirection = -tempDirection;
                }

                tempDirection = tempDirection + norm;

                tempDirection.Normalize();
                // tempDirection = tempDirection * 5f;

                Direction = tempDirection;

            }


        }
        
        
        return Direction;
 
    }

    Vector3 get_away_from_wall(List<GameObject> wall_list,Vector3 Direction)
    {
        if (wall_list.Count == 1)
        {
            // Debug.Log("HEre");
            Vector3 wall_normal = wall_list[0].transform.forward.normalized;
            
            if (Vector3.Dot(wall_normal,Direction.normalized) >=0.95f)
            {
                // Debug.Log("HEre1");
                Direction = rotate(Direction,Mathf.PI/4f,false);
            }
            else if (Vector3.Dot(wall_normal,Direction.normalized) <= - 0.95f)
            {
                // Debug.Log("HEre2");
                Direction = rotate(Direction,Mathf.PI/4f,true);
            }
        }
        return Direction;
    }

    bool IsPositionWalkable(Vector3 position)
    {
        // UnityEngine.AI.NavMeshHit hit;
        bool in_range  = (position.x > -23 && position.x < 23) && (position.z > -23 && position.z < 23);
        return in_range;
    }

    bool NoObstacle(Transform myself, Vector3 direction, float rayLength)
    {
        // Cast a ray from the 'myself' position in the specified direction
        RaycastHit hit;
        if (Physics.Raycast(myself.position, direction, out hit, rayLength))
        {
            // If the ray hits something, check if the hit object has the tag "Obstacle"
            if (hit.collider.CompareTag("Obstacle") || hit.collider.CompareTag("Hider"))
            {
                // If it's tagged as an obstacle, return false
                return false;
            }
        }
        
        // If no obstacles are hit or if the hit object is not tagged as an obstacle, return true
        return true;
    }

    bool NoWall(Transform myself, Vector3 direction, float rayLength)
    {
        // Cast a ray from the 'myself' position in the specified direction
        RaycastHit hit;
        if (Physics.Raycast(myself.position, direction, out hit, rayLength))
        {
            // If the ray hits something, check if the hit object has the tag "Obstacle"
            if (hit.collider.CompareTag("Wall"))
            {
                // If it's tagged as an obstacle, return false
                return false;
            }
        }
        
        // If no obstacles are hit or if the hit object is not tagged as an obstacle, return true
        return true;
    }


    Vector3 Getangle(float theta, float Distance1, float Distance2, Vector3 B)
    {
        float theta2 = (2 * Mathf.PI - theta) * Distance1/ (Distance1 + Distance2);

        float sinAngle = Mathf.Sin(theta2);
        float cosAngle = Mathf.Cos(theta2);

        float newX = B.x * cosAngle - B.z * sinAngle;
        float newZ = B.x * sinAngle + B.z * cosAngle;
        Vector3 Heading = new Vector3(newX, 0, newZ);

        return Heading;
    }


    Vector3 rotate(Vector3 oD,float theta, bool clockwise)
    {
        float sinAngle = Mathf.Sin(theta);
        float cosAngle = Mathf.Cos(theta);
        float newX;
        float newZ;
        if (clockwise == false)
        {
            newX = oD.x * cosAngle - oD.z * sinAngle;
            newZ = oD.x * sinAngle + oD.z * cosAngle;
        }
        else
        {
            newX = oD.x * cosAngle + oD.z * sinAngle;
            newZ = -oD.x * sinAngle + oD.z * cosAngle;
        }
        Vector3 Heading = new Vector3(newX, 0, newZ);

        return Heading;
    }

    Vector3 calculate_run_away_direction(Transform myself, List<GameObject> target_list, List<GameObject> wall_list) //need to generalized to more than 2 seeker case
    {
        if (target_list.Count > 1 && wall_list.Count != 0)
        {
            // handle this in a different function later
            return myself.forward;
        }
        else //no wall around
        {
            if (target_list.Count == 0) //no seeker
            {
                // keep going into the current current direction
                return myself.forward;
            }
            else if (target_list.Count == 1) // one seeker
            {
                if (wall_list.Count > 0)
                {
                    special_case = true;
                }
                
                Vector3 Direction = myself.position - target_list[0].transform.position;
                Direction.y = 0f;
                return myself.position - target_list[0].transform.position;
            }
            else // multiple seekers  //Generalize
            {
                special_case = false;
                
                // Debug.Log("Old Target List: [" + string.Join(", ", target_list) + "]");
                
                result_list = check_seeker_all_on_one_side(target_list,transform);
                // Debug.Log("Old Target List: [" + string.Join(", ", result_list) + "]");
                
                //if there are all on one side, then pick the two with the biggest inner angle as the two target to run away
                if (result_list[0] == 1)
                {
                    
                    if (result_list.Count > 1)
                    {
                        List<GameObject> target_list_copy = new List<GameObject>();
                        target_list_copy.Add(target_list[result_list[1]]);
                        target_list_copy.Add(target_list[result_list[2]]);
                        target_list = target_list_copy; 
                    }

                    // Debug.Log("New Target List: [" + string.Join(", ", target_list) + "]");
                    var target1 = target_list[0];
                    var target2 = target_list[1];

                    var A= target1.transform.position - transform.position;
                    var B = target2.transform.position - transform.position;

                    var Distance1 = A.magnitude;
                    var Distance2 = B.magnitude;

                    float theta = Mathf.Atan2(Vector3.Magnitude(Vector3.Cross(A , B)), Vector3.Dot(A , B ));
                    
                    float slope = (target1.transform.position.z - target2.transform.position.z) / (target1.transform.position.x - target2.transform.position.x);
                    float intersection = target1.transform.position.z - slope * target1.transform.position.x;
                    float threshold = slope * myself.position.x + intersection;

                    Vector3 Direction;
                    if (threshold > myself.position.z)
                    {
                        if (target1.transform.position.x <= target2.transform.position.x)
                        {
                            Direction = Getangle(theta, Distance2, Distance1, A);
                        }
                        else
                        {
                            Direction = Getangle(theta, Distance1, Distance2, B);
                        }
                    }
                    else
                    {
                        if (target1.transform.position.x >= target2.transform.position.x)
                        {
                            Direction = Getangle(theta, Distance2, Distance1, A);
                        }
                        else
                        {
                            Direction = Getangle(theta, Distance1, Distance2, B);
                        }
                    }
                    // Debug.Log("Running Direction:"+Direction);

                    Direction.y = 0f;
                    return Direction;
                }
                else //if they are sourounding the hider then pick the biggest gap to run from the middle
                {
                    Direction = run_in_middle(wall_list,target_list,Direction);

                    // Debug.Log(Direction);
                    Direction.y = 0f;
                    is_run_in_middle = true;
                    special_case = false;
                    
                    keep_action_counter = 0;
                    return Direction;
                }
        
                
            }
        }
        
    }


    List<int> check_seeker_all_on_one_side(List<GameObject> target_list, Transform transform)
    {
        List<int> result = new List<int>();
        if (target_list.Count <= 2) //less than or equal 2 seekers
        {
            result.Add(1);
            return result;
        }
        else //more than 2 seekers
        {
            // Debug.Log("This place");
            List<Vector2> vectorList = new List<Vector2>();
            foreach (var seeker in target_list)
            {
                vectorList.Add(new Vector2(seeker.transform.position.x-transform.position.x, seeker.transform.position.z - transform.position.z)); 
            }
            
            
            List<float> angleList = new List<float>();
            float maxangle;
            int indexA;
            int indexB;           

            
            int positiveCrosses = 0;
            int negativeCrosses = 0;

            bool all_seeker_same_side = false;

            foreach (var referenceVector in vectorList)
            {
                foreach (var vector in vectorList)
                {
                    // Compute the cross product of the reference vector and the current vector
                    float crossProduct = referenceVector.x * vector.y - referenceVector.y * vector.x;

                    if (crossProduct > 0)
                    {
                        positiveCrosses++;
                    }
                    else if (crossProduct < 0)
                    {
                        negativeCrosses++;
                    }

                }

                all_seeker_same_side = !(positiveCrosses > 0 && negativeCrosses > 0);
                if (all_seeker_same_side)
                {
                    break;
                }

                positiveCrosses = 0;
                negativeCrosses = 0;
            }

            if (all_seeker_same_side)
            {
                //comvert to < 180 angle
                vectorList = new List<Vector2>();
                foreach (var seeker in target_list)
                {
                    vectorList.Add(new Vector2(seeker.transform.position.x-transform.position.x, seeker.transform.position.z - transform.position.z)); 
                }
                
                
                angleList = new List<float>();
                maxangle = 0f;
                indexA = 0;
                indexB = 0;           

                for (int i = 0; i < vectorList.Count; i++)
                {
                    for (int j = i + 1; j < vectorList.Count; j++)
                    {
                        float angle = Vector2.Angle(vectorList[i], vectorList[j]);
                        angleList.Add(angle);
                        if (angle >= maxangle)
                        {
                            maxangle = angle;
                            indexA = i;
                            indexB = j;
                        }
                    }
                }

                result.Add(1);
                result.Add(indexA);
                result.Add(indexB);
                return result; 
            }
            else //not all on the same side
            {
                vectorList = new List<Vector2>();
                foreach (var seeker in target_list)
                {
                    vectorList.Add(new Vector2(seeker.transform.position.x-transform.position.x, seeker.transform.position.z - transform.position.z)); 
                }
                
                
                maxangle = 0f;
                indexA = -1;
                indexB = -1;           

                for (int i = 0; i < vectorList.Count; i++)
                {
                    for (int j = i + 1; j < vectorList.Count; j++)
                    {
                        float angle = Vector2.Angle(vectorList[i], vectorList[j]);

                        if (angle >= maxangle && check_adjacent(vectorList,i,j)&&not_both_seeker_next_to_wall(target_list,i,j))
                        {
                            maxangle = angle;
                            indexA = i;
                            indexB = j;
                        }
                    }
                }

                if (indexA == indexB)
                {
                    float least_distance1 = Mathf.Infinity;
                    float least_distance2 = Mathf.Infinity;

                    for (int i = 0; i < vectorList.Count; i++)
                    {
                        float distance_to_walls = Mathf.Abs(target_list[i].transform.position.x) + Mathf.Abs(target_list[i].transform.position.z);

                        if (distance_to_walls < least_distance1 && distance_to_walls < least_distance2)
                        {
                            indexB = indexA;
                            indexA = i;
                            least_distance2 = least_distance1;
                            least_distance1 = distance_to_walls;
                            
                        }
                        else if (distance_to_walls >=  least_distance1 && distance_to_walls < least_distance2)
                        {
                            indexB = i;
                            least_distance2 = distance_to_walls;
                        }
                        else
                        {
                            
                        }

                        Debug.Log(indexA+","+indexB);
                        Debug.Log(least_distance1+","+least_distance2);


                    }
                }

                result.Add(0);
                result.Add(indexA);
                result.Add(indexB);
                return result;
            }         
        }
    }

    float CalculateClockwiseAngle(Vector2 from, Vector2 to, bool clockwise)
    {
        // Normalize the vectors
        from.Normalize();
        to.Normalize();

        // Calculate the dot product
        float dot = Vector2.Dot(from, to);

        // Calculate the angle in radians
        float angle = Mathf.Acos(dot);

        // Convert to degrees
        angle = angle * Mathf.Rad2Deg;

        // Calculate the cross product
        float cross = from.x * to.y - from.y * to.x;

        // Determine if the angle is clockwise or counterclockwise
        if (clockwise)
        {
            if (cross < 0)
            {
                angle = 360f - angle;
            }
        }
        else
        {
            if (cross >= 0)
            {
                angle = 360f - angle;
            }
        }

        return angle;
    }

    bool check_adjacent(List<Vector2> vectorList, int i, int j)
    {
        bool result = true;
        for (int m = 0; m < vectorList.Count; m++)
        {
            if (m != i &&  m != j)
            {
                float angle1 = Vector2.Angle(vectorList[i],vectorList[j]);
                float angle2 = Vector2.Angle(vectorList[i],vectorList[m]);
                float angle3 = Vector2.Angle(vectorList[m],vectorList[j]);

                if (Mathf.Approximately(angle1,angle2+angle3))
                {
                    result = false;
                    break;
                }
            }
        }

        return result;

    }

    List<int> check_seeker_all_on_one_side_wall(List<GameObject> target_list,GameObject wall, Transform transform)
    {
        List<int> result = new List<int>();
        if (target_list.Count < 2) //less than 2 seekers
        {
            result.Add(1);
            return result;
        }
        else //at least 2 seekers
        {
            List<Vector2> vectorList = new List<Vector2>();
            foreach (var seeker in target_list)
            {
                vectorList.Add(new Vector2(seeker.transform.position.x-transform.position.x, seeker.transform.position.z - transform.position.z)); 
            }
            
            Vector3 wall_norm = rotate(wall.transform.forward,Mathf.PI/2,true); 

            Vector2 referenceVector = new Vector2(wall_norm.x,wall_norm.z);
            
            int positiveCrosses = 0;
            int negativeCrosses = 0;

            // bool all_seeker_same_side = false;

            foreach (var vector in vectorList)
            {
                // Compute the cross product of the reference vector and the current vector
                float crossProduct = referenceVector.x * vector.y - referenceVector.y * vector.x;

                if (crossProduct > 0)
                {
                    positiveCrosses++;
                }
                else if (crossProduct < 0)
                {
                    negativeCrosses++;
                }

            }

            bool all_seeker_same_side = !(positiveCrosses > 0 && negativeCrosses > 0);

            if (all_seeker_same_side)
            {
                result.Add(1);
                return result; 
            }
            else //not all on the same side
            {
                            
                List<float> gapList = new List<float>();
                float maxgap;
                int indexA;
                int indexB;                           
                
                maxgap = 0f;
                indexA = 0;
                indexB = 0;           

                for (int i = 0; i < vectorList.Count; i++)
                {
                    for (int j = i + 1; j < vectorList.Count; j++)
                    {
                        float gap = Vector2.Distance(vectorList[i], vectorList[j]);

                        if (gap >= maxgap && check_adjacent(vectorList,i,j) && check_wall_in_middle(vectorList,i,j,wall)) 
                        {
                            maxgap = gap;
                            indexA = i;
                            indexB = j;
                        }
                    }

                    float wall_gap = 25f - Mathf.Abs(Vector3.Dot(target_list[i].transform.position,rotate(wall.transform.forward,Mathf.PI/2,true)));

                    if (wall_gap >= maxgap && check_adjacent_wall(target_list,vectorList,i,wall))
                    {
                        maxgap = wall_gap;
                        indexA = i;
                        indexB = -1;
                    }
                }

                result.Add(0);
                result.Add(indexA);
                result.Add(indexB);
                return result;
            }         
        }
    }

    bool check_adjacent_wall(List<GameObject> target_list,List<Vector2> vectorList,int i, GameObject wall)
    {
        bool result1 = true;
        Vector2 wall_Direction = new Vector2(wall.transform.forward.x,wall.transform.forward.z);
        for (int m =0; m < vectorList.Count;m++)
        {
            if (m != i)
            {
                if ((Vector2.Dot(wall_Direction,vectorList[i])*Vector2.Dot(wall_Direction,vectorList[m])) >= 0f)
                {
                    float current_gap =  25f - Mathf.Abs(Vector3.Dot(target_list[i].transform.position,rotate(wall.transform.forward,Mathf.PI/2,true)));
                    float compare_gap =  25f - Mathf.Abs(Vector3.Dot(target_list[m].transform.position,rotate(wall.transform.forward,Mathf.PI/2,true)));
                    
                    if (current_gap > compare_gap)
                    {
                        result1 = false;
                        break;
                    }
                }
            }
        }

        return result1;
    }

    bool check_wall_in_middle(List<Vector2> vectorList,int i,int j,GameObject wall) //check if the two seekers are both and the end and the wall is in between
    {
        Vector3 wall_direction = rotate(wall.transform.forward, Mathf.PI/2,true);
        Vector2 wall_norm = new Vector2(wall_direction.x,wall_direction.z);

        float angle1 = Vector2.Angle(vectorList[i],vectorList[j]);
        float angle2 = Vector2.Angle(vectorList[i],wall_norm);
        float angle3 = Vector2.Angle(vectorList[j],wall_norm);

        if (Mathf.Approximately(angle1,angle2+angle3))
        {
            return false;
        }
        else
        {
            return true;
        }
        
    }


    List<int> check_seeker_all_on_one_side_corner(List<GameObject> target_list, List<GameObject> wall_list, Transform transform)
    {
        List<int> result = new List<int>();
        if (target_list.Count < 2) //less than 2 seekers
        {
            result.Add(1);
            return result;
        }
        else //at least 2 seekers
        {
            List<Vector2> vectorList = new List<Vector2>();
            foreach (var seeker in target_list)
            {
                vectorList.Add(new Vector2(seeker.transform.position.x-transform.position.x, seeker.transform.position.z - transform.position.z)); 
            }
            
            Vector3 wall_norm = rotate(wall_list[0].transform.forward+wall_list[1].transform.forward,Mathf.PI/2,false); 

            Vector2 referenceVector = new Vector2(wall_norm.x,wall_norm.z);
            
            int positiveCrosses = 0;
            int negativeCrosses = 0;

            // bool all_seeker_same_side = false;

            foreach (var vector in vectorList)
            {
                // Compute the cross product of the reference vector and the current vector
                float crossProduct = referenceVector.x * vector.y - referenceVector.y * vector.x;

                if (crossProduct > 0)
                {
                    positiveCrosses++;
                }
                else if (crossProduct < 0)
                {
                    negativeCrosses++;
                }

            }

            bool all_seeker_same_side = !(positiveCrosses > 0 && negativeCrosses > 0);

            if (all_seeker_same_side) //on one side
            {
                result.Add(1);
                return result; 
            }
            else //not all on the same side
            {
                            
                List<float> gapList = new List<float>();
                float maxgap;
                int indexA;
                int indexB;                           
                
                maxgap = 0f;
                indexA = 0;
                indexB = 0;           

                for (int i = 0; i < vectorList.Count; i++)
                {
                    for (int j = i + 1; j < vectorList.Count; j++)
                    {
                        float gap = Vector2.Distance(vectorList[i], vectorList[j]);

                        if (gap >= maxgap && check_adjacent(vectorList,i,j) && check_wall_in_middle(vectorList,i,j,wall_list[0]) && check_wall_in_middle(vectorList,i,j,wall_list[1])) 
                        {
                            maxgap = gap;
                            indexA = i;
                            indexB = j;
                        }
                    }

                    GameObject wall1 = wall_list[0];
                    GameObject wall2 = wall_list[1];
                    GameObject wall;

                    
                    float wall_gap1 = 25f - Mathf.Abs(Vector3.Dot(target_list[i].transform.position,rotate(wall1.transform.forward,Mathf.PI/2,true)));
                    float wall_gap2 = 25f - Mathf.Abs(Vector3.Dot(target_list[i].transform.position,rotate(wall2.transform.forward,Mathf.PI/2,true)));
                    
                    float wall_gap = Mathf.Min(wall_gap1,wall_gap2); 
                    int tempind;

                    if (wall_gap1 > wall_gap2)
                    {
                        wall = wall2;
                        tempind = -2;
                    }else
                    {
                        wall = wall1;
                        tempind = -1;
                    }

                    if (wall_gap >= maxgap && check_adjacent_wall(target_list,vectorList,i,wall))
                    {

                        maxgap = wall_gap;
                        indexA = i;
                        indexB = tempind;
                    }


                }

                result.Add(0);
                result.Add(indexA);
                result.Add(indexB);
                return result;
            }         
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Hider"))
        {
            Physics.IgnoreCollision(collision.collider, GetComponent<Collider>());
        }
    }

    bool new_seeker_in_range(List<GameObject> last_target_list ,List<GameObject> target_list)
    {
        bool result = false;
        foreach(var target in target_list)
        {
            if (!last_target_list.Contains(target))
            {
                result = true;
                break;
            }

        }

        return result;
    }

    bool not_both_seeker_next_to_wall(List<GameObject> target_list, int i, int j)
    {
        var target1 = target_list[i];
        var target2 = target_list[j];

        return !((Mathf.Abs(target1.transform.position.x) > 18f || Mathf.Abs(target1.transform.position.z) > 18f)&&
        (Mathf.Abs(target2.transform.position.x) > 18f || Mathf.Abs(target2.transform.position.z) > 18f));
    }

}
}