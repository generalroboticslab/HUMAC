using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


namespace Examples.HideAndSeek
{
public class SeekerHeruistic : MonoBehaviour
{
    public int catch_hider;
    public float maxspeed;
    private float speed;

    private GameManager gm;
    private Collider[] hitcolliders;

    public float detectrange;
    public Rigidbody rb;
    public GameObject target;

    public List<GameObject> target_list = new List<GameObject>();

    public bool seeplayer;
    Vector3 LasthiderPosition = new Vector3(float.NaN, float.NaN, float.NaN);
    Vector3 nullVector = new Vector3(float.NaN, float.NaN, float.NaN);
    Vector3 Heading;
    private float counter;

    private Vector3 lastDirection;
    
    public List<GameObject> wall_list = new List<GameObject>();
    public List<int> wall_number_list =new List<int> {0, 0};

    float angle;
    bool cw;
    GameObject old_wall = null;
    GameObject wall;

    public float walldetectrange;
    public float obstacledetectrange;
    //public Vector3 Initial_direction;

    private Vector3 in_direction;

    private bool corner_turn = false;
    float counter1 = 0f;
    float counter2 = 0f;
    float first_wall_angle;
    float tt_angle = 0f;
    bool see_obstacle = false;
    RaycastHit hit;
    public float distance;
    // Start is called before the first frame update
    bool ob_turn = false;

    private Navigation navMeshAgent;
    void Start()
    {
        navMeshAgent = GetComponent<Navigation>();

        speed = maxspeed;
        counter = 0;
        gm = FindObjectOfType<GameManager>();

        lastDirection = transform.forward;
        lastDirection.Normalize();

    }

    // Update is called once per frame
    void FixedUpdate()
    {
        lastDirection = transform.forward;
        if (!navMeshAgent.enabled)
    {
        distance = 1000f;
        if (!gm.GameRunning || gm.GamePaused)
        {
            speed = 0f;
        }else
        {
            speed = maxspeed;
        }

        see_obstacle = false;
        counter = counter + Time.deltaTime;
        Heading = lastDirection;
        //detect hider in range
        seeplayer = false;

        target_list.Clear();
        Collider[] hitColliders = Physics.OverlapBox(transform.position, new Vector3(detectrange,detectrange,detectrange), Quaternion.identity);
        foreach (Collider col in hitColliders)
        {
            if (col.CompareTag("Hider"))
            {
                target_list.Add(col.gameObject);
                // target = col.gameObject;
                seeplayer = true;

            }
            
        }

        if (target_list.Count > 0)
        {
            float distance = Mathf.Infinity;
            foreach (GameObject hider in target_list)
            {
                if (Vector3.Distance(hider.transform.position,transform.position) < distance)
                {
                    target = hider;
                    distance = Vector3.Distance(hider.transform.position,transform.position);
                }
            }
        }
            

        //detect obstacle
        Vector3 left_d1 = rotate(transform.forward, Mathf.PI/3,false);
        for (int i = 0; i <= 120; i++)
        {
            Vector3 ray_d = rotate(left_d1,Mathf.PI/180*i,true);
            if (Physics.Raycast(transform.position,ray_d,out hit,obstacledetectrange))
            {
                if (hit.collider.CompareTag("Obstacle"))
                {
                    see_obstacle = true;
                    break;
                }
            }
        }


        //No hider in range
        if (!seeplayer) //didn't see hider
        {
            //detect wall in range
            wall_list.Clear();
            RaycastHit[] hits = Physics.SphereCastAll(new Ray(transform.position, Vector3.up),walldetectrange);
            foreach (var hit in hits)
            {
                if (hit.collider.CompareTag("Wall"))
                {
                    wall_list.Add(hit.collider.gameObject);
                }
            }

            wall_number_list[0] = wall_number_list[1];
            wall_number_list[1] = wall_list.Count;
            
            if (see_obstacle)
            {
                ob_turn = false;
                for (int i = 0; i < 6;i++)
                {
                    Vector3 ray_d11 = rotate(transform.forward,Mathf.PI/180*10f*i,true);
                    Vector3 ray_d12 = rotate(transform.forward,Mathf.PI/180*10f*(i+1),true);
                    Vector3 ray_d13 = rotate(transform.forward,Mathf.PI/180*10f*(i-1),true);
                    if (!Physics.Raycast(transform.position, ray_d11, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d12, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d13, out hit, obstacledetectrange)
                    )
                    {
                        Heading = ray_d11;
                        ob_turn = true;
                        break;
                    }
                    
                    Vector3 ray_d21 = rotate(transform.forward,Mathf.PI/180*10f*i,false);
                    Vector3 ray_d22 = rotate(transform.forward,Mathf.PI/180*10f*(i+1),false);
                    Vector3 ray_d23 = rotate(transform.forward,Mathf.PI/180*10f*(i-1),false);
                    if (!Physics.Raycast(transform.position, ray_d21, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d22, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d23, out hit, obstacledetectrange)
                    )
                    {
                        Heading = ray_d21;
                        ob_turn = true;
                        break;
                    }
                }

                if (!ob_turn)
                {
                    Heading = rotate(transform.forward,Mathf.PI/3,true);
                }
            }

            //Initialize current direction
            if (float.IsNaN(LasthiderPosition.x)) //No hider's last direction
            {
                //wall around
                if (wall_number_list[1] != 0)
                {
                    //open space to corner
                    if (wall_number_list[1] - wall_number_list[0] == 2) 
                    {
                        tt_angle = 0.0f;
                        corner_turn = true;

                        //pick the first wall as the reference to turn.
                        wall = wall_list[0];

                        Vector3 cD =  transform.forward;
                        Vector3 wD = wall.transform.forward;
                        float total_angle = Mathf.Atan2(Vector3.Magnitude(Vector3.Cross(cD, wD)), Vector3.Dot(cD, wD));
                        in_direction = cD;

                        //get the angle and direction correct
                        if (total_angle >= Mathf.PI/2)
                        {
                            cw = false;
                        }
                        else
                        {
                            total_angle = Mathf.PI - total_angle; 
                            cw = true;
                        }



                        if (total_angle != 0 && total_angle != Mathf.PI)
                        {
                            float d_to_wall = walldetectrange/1.1f;
                            float time = d_to_wall/(speed*Time.deltaTime);
                            float delta_angle = total_angle/time;

                            angle = delta_angle;
                            Heading =  rotate(lastDirection,delta_angle,cw);
                        }

                        old_wall = wall;
                        
                    }

                    //run into an extra wall
                    else if (wall_number_list[1] - wall_number_list[0] == 1)
                    {
                        tt_angle = 0.0f;
                        //open space to wall
                        if (wall_number_list[0] == 0)
                        {
                            wall = wall_list[0];
                            corner_turn = false;
                            counter1 = counter;
                            first_wall_angle = Mathf.Atan2(Vector3.Magnitude(Vector3.Cross(transform.forward, wall.transform.forward)), Vector3.Dot(transform.forward, wall.transform.forward));    
                        }
                        //wall to corner
                        else
                        {
                            counter2 = counter;
                            corner_turn = true;
                            foreach (GameObject wallob in wall_list)
                                if (wallob != old_wall)
                                {
                                    wall = wallob;
                                }
                        }
                        
                        //consider running into a corner from open space
                        if (counter2-counter1 > 0 && counter2-counter1 <= 1f )
                        {
                            //run into corner
                            corner_turn = true;
                            //pick the first wall as the reference to turn.
                            
                            Vector3 cD =  transform.forward;
                            Vector3 wD = wall.transform.forward;
                            float total_angle = Mathf.Atan2(Vector3.Magnitude(Vector3.Cross(cD, wD)), Vector3.Dot(cD, wD));
                            in_direction = cD;
                            //get the angle and direction correct
                            
                            if (first_wall_angle >= Mathf.PI/2)
                                if (total_angle >= Mathf.PI/2)
                                {
                                    total_angle = Mathf.PI - total_angle;
                                    cw = true;

                                }
                                else
                                {
                                    total_angle = Mathf.PI - total_angle; 
                                    cw = true;
                                }
                            else
                            {
                                cw = false;
                            }
                            

                            if (total_angle != 0 && total_angle != Mathf.PI)
                            {
                                float d_to_wall = walldetectrange/2f;
                                float time = d_to_wall/(speed*Time.deltaTime);
                                float delta_angle = total_angle/time;

                                angle = delta_angle;
                                Heading =  rotate(lastDirection,delta_angle,cw);
                            }

                            old_wall = wall;
                        }

                        //from open to wall
                        else
                        {
                            corner_turn = false;

                            Vector3 cD =  transform.forward;
                            Vector3 wD = wall.transform.forward;
                            float total_angle = Mathf.Atan2(Vector3.Magnitude(Vector3.Cross(cD, wD)), Vector3.Dot(cD, wD));
                            float delta_angle;
                            in_direction = cD;


                            if (total_angle >= Mathf.PI/2)
                            {
                                total_angle = Mathf.PI - total_angle; 
                                cw = true;
                            }
                            else
                            {
                                total_angle = Mathf.PI - total_angle; 
                                cw = false;
                            }
                            

                            if (total_angle != 0 && total_angle != Mathf.PI)
                            {
                                float d_to_wall = walldetectrange;
                                float time = d_to_wall/(speed*Time.deltaTime);
                                delta_angle = total_angle/time;
                                angle = delta_angle;
                                Heading = rotate(lastDirection,delta_angle,cw);                           
                            }

                            old_wall = wall;
                        }
                    }
                    
                    //still in wall or corner
                    else
                    {
                        //still in wall or corner
                        Vector3 v1 = transform.forward;
                        if (corner_turn)
                        {
                            if (Vector3.Dot(in_direction, v1) <= -0.97f)
                            {
                                angle = 0.0f;
                            }else
                            {
                                Heading = rotate(lastDirection,angle,cw);                      
                            }
                        }
                        else
                        {
                            Heading = rotate(lastDirection,angle,cw);    
                        }
                        
                        tt_angle += angle;

                        if (tt_angle >= Mathf.PI)
                        {
                            angle = 0f;
                        }

                    }
                }
                
            }

        else //Know hider's last direction, move to hider's last show up position
            {
                Heading = LasthiderPosition - transform.position;
                if (wall_list.Count == 0) 
                {
                    if (Heading.x <= 0.1f && Heading.z <= 0.1f)
                    {
                        LasthiderPosition =nullVector;
                        Heading = transform.forward;
                        //Heading.y = 0;
                    } 
                }
                else
                {
                    LasthiderPosition =nullVector;
                    wall = wall_list[0];
                    if (Vector3.Dot(transform.forward,wall.transform.forward) >= 0.0f)
                    {
                        Heading = rotate(lastDirection,Mathf.PI/10,false);
                    } 
                    else
                    {
                        Heading = rotate(lastDirection,Mathf.PI/10,true);
                    }



                }
                
                if (see_obstacle)
                {
                    ob_turn = false;
                    for (int i = 0; i < 6;i++)
                    {
                        Vector3 ray_d11 = rotate(transform.forward,Mathf.PI/180*10f*i,true);
                        Vector3 ray_d12 = rotate(transform.forward,Mathf.PI/180*10f*(i+1),true);
                        Vector3 ray_d13 = rotate(transform.forward,Mathf.PI/180*10f*(i-1),true);
                        if (!Physics.Raycast(transform.position, ray_d11, out hit, obstacledetectrange)
                        && !Physics.Raycast(transform.position, ray_d12, out hit, obstacledetectrange)
                        && !Physics.Raycast(transform.position, ray_d13, out hit, obstacledetectrange)
                        )
                        {
                            Heading = ray_d11;
                            ob_turn = true;
                            break;
                        }
                        
                        Vector3 ray_d21 = rotate(transform.forward,Mathf.PI/180*10f*i,false);
                        Vector3 ray_d22 = rotate(transform.forward,Mathf.PI/180*10f*(i+1),false);
                        Vector3 ray_d23 = rotate(transform.forward,Mathf.PI/180*10f*(i-1),false);
                        if (!Physics.Raycast(transform.position, ray_d21, out hit, obstacledetectrange)
                        && !Physics.Raycast(transform.position, ray_d22, out hit, obstacledetectrange)
                        && !Physics.Raycast(transform.position, ray_d23, out hit, obstacledetectrange)
                        )
                        {
                            Heading = ray_d21;
                            ob_turn = true;
                            break;
                        }
                    }

                    if (!ob_turn)
                    {
                        Heading = rotate(transform.forward,Mathf.PI/3,true);
                    }
                }
            }
        }

        else // see the hider. Go chase it.
        {
            Heading = target.transform.position - transform.position;
            if (see_obstacle)
            {
                ob_turn = false;
                for (int i = 0; i < 6;i++)
                {
                    Vector3 ray_d11 = rotate(transform.forward,Mathf.PI/180*10f*i,true);
                    Vector3 ray_d12 = rotate(transform.forward,Mathf.PI/180*10f*(i+1),true);
                    Vector3 ray_d13 = rotate(transform.forward,Mathf.PI/180*10f*(i-1),true);
                    if (!Physics.Raycast(transform.position, ray_d11, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d12, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d13, out hit, obstacledetectrange)
                    )
                    {
                        Heading = ray_d11;
                        ob_turn = true;
                        break;
                    }
                    
                    Vector3 ray_d21 = rotate(transform.forward,Mathf.PI/180*10f*i,false);
                    Vector3 ray_d22 = rotate(transform.forward,Mathf.PI/180*10f*(i+1),false);
                    Vector3 ray_d23 = rotate(transform.forward,Mathf.PI/180*10f*(i-1),false);
                    if (!Physics.Raycast(transform.position, ray_d21, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d22, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d23, out hit, obstacledetectrange)
                    )
                    {
                        Heading = ray_d21;
                        ob_turn = true;
                        break;
                    }
                }

                if (!ob_turn)
                {
                    Heading = rotate(transform.forward,Mathf.PI/3,true);
                }
            }
            LasthiderPosition = target.transform.position ;


        }

        //normalize the direction and move the cube.
        Heading.y = 0;
        Heading.Normalize();
        if (speed != 0)
        {
            Vector3 Move = new Vector3(Heading.x*speed,0,Heading.z*speed);
            rb.velocity = Move;
            transform.forward  =  Move;
        }
        //save the last direction
        lastDirection = Heading;
    }
    }

 // Collision detection with the Hider
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Hider"))
        {
            catch_hider++;
        }

        if (collision.gameObject.CompareTag("Wall") )
        {
            Vector3 normal = collision.contacts[0].normal;
            Heading = lastDirection - 2*Vector3.Dot(normal,lastDirection)*normal ;
            lastDirection = Heading;
        }

        if (collision.gameObject.CompareTag("Obstacle"))
        {
            if (!seeplayer && LasthiderPosition == nullVector)
            {
                Vector3 normal = collision.contacts[0].normal;
                Heading = lastDirection - 2*Vector3.Dot(normal,lastDirection)*normal ;
                lastDirection = Heading;
            }
            else
            {
                bool turned = false;

                for (int i = 0; i < 10;i++)
                {
                    Vector3 ray_d11 = rotate(transform.forward,Mathf.PI/180*9f*i,true);
                    Vector3 ray_d12 = rotate(transform.forward,Mathf.PI/180*9f*(i+1),true);
                    Vector3 ray_d13 = rotate(transform.forward,Mathf.PI/180*9f*(i-1),true);
                    if (!Physics.Raycast(transform.position, ray_d11, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d12, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d13, out hit, obstacledetectrange)
                    )
                    {
                        Heading = ray_d11;
                        turned = true;
                        break;
                    }
                    
                    Vector3 ray_d21 = rotate(transform.forward,Mathf.PI/180*9f*i,false);
                    Vector3 ray_d22 = rotate(transform.forward,Mathf.PI/180*9f*(i+1),false);
                    Vector3 ray_d23 = rotate(transform.forward,Mathf.PI/180*9f*(i-1),false);
                    if (!Physics.Raycast(transform.position, ray_d21, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d22, out hit, obstacledetectrange)
                    && !Physics.Raycast(transform.position, ray_d23, out hit, obstacledetectrange)
                    )
                    {
                        Heading = ray_d21;
                        turned = true;
                        break;
                    }
                }
                if (!turned)
                {

                    Heading = rotate(Heading, Mathf.PI/2,true);
                }
                    
            }
            
        }

        if (collision.gameObject.CompareTag("Seeker"))
        {   
            if (!seeplayer)
            {
                if (LasthiderPosition == nullVector)
                {
                    lastDirection = -Heading;
                }
            }
        }
    }
    
    //rotation function
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

}

}