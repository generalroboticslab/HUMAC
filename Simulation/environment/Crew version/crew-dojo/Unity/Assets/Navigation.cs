using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Navigation : MonoBehaviour
{
    [SerializeField]
    public string ObstacleTag = "Obstacle";

    public Transform body;

    [SerializeField]
    public float speed = 5.0f;

    [SerializeField]
    public float obstacle_detection_range = 5.0f;

    private bool has_destination = false;

    private Vector3 destination;

    private float Agent_radius;

    private float rotation_angle;

    void Start()
    {
        //get the body from the object the scrip t attached to
        body = GetComponent<Transform>();
        // Debug.Log(body);
        if (body == null)
        {
            Debug.LogError("No body found");
        }
        Agent_radius = body.localScale.x / 2;

        rotation_angle = Mathf.Atan(Agent_radius/obstacle_detection_range); 
        rotation_angle = rotation_angle * Mathf.Rad2Deg; // Convert radians to degrees if needed
        //convert to int by up round
        rotation_angle = Mathf.Ceil(rotation_angle);
        //make sure 180 is divisible by rotation_angle
        while (180 % rotation_angle != 0)
        {
            rotation_angle += 1;
        }

        // Debug.Log("rotation:"+rotation_angle);
        // Debug.Log("Agent_radius: " + Agent_radius);
    }

    void FixedUpdate()
    {
        if (has_destination)
        {
            if (! reachedDestination())
            {
                
                
                Vector3 direction = destination - body.position;
                direction.y = 0f;
                
                direction = AvoidObstacle(direction,rotation_angle);
                direction.Normalize();
                
                body.forward = direction;
                transform.position += direction * speed * Time.deltaTime;
            }
            else
            {
                Stop();
            }
        }
    }

    public Vector3 AvoidObstacle(Vector3 Direction, float rotation_angle)
    {

        //cast rotationangle to integer
        float num = 180/rotation_angle;
        // Debug.Log(num);

        for (int j = 0;j <= num*2;j++)
        {
            
            Vector3 ray_d11 = rotate(Direction,Mathf.PI/180*rotation_angle*j,true); // change this angle
            Vector3 ray_d12 = rotate(Direction,Mathf.PI/180*rotation_angle*(j+1),true); // change this angle
            Vector3 ray_d13 = rotate(Direction,Mathf.PI/180*rotation_angle*(j-1),true); // change this angle

            if (NoObstacle(body,ray_d11,obstacle_detection_range)  
            &&  NoObstacle(body,ray_d12,obstacle_detection_range) 
            && NoObstacle(body,ray_d13,obstacle_detection_range))
            {
                //print(j);

                Direction = ray_d11;
                break;
            }
        }

        return Direction;
    }

    private bool NoObstacle(Transform myself, Vector3 direction, float rayLength)
    {
        // Debug.Log("myself: " + myself);
        // Cast a ray from the 'myself' position in the specified direction
        RaycastHit hit;
        if (Physics.Raycast(myself.position, direction, out hit, rayLength))
        {
            if (hit.collider.CompareTag("Obstacle"))
            {
                return false;
            }
        }
        return true;
    }

    public void SetDestination(Vector3 destination)
    {
        this.destination = destination;
        has_destination = true;
    } 

    public void Stop()
    {
        has_destination = false;
        //forget the destination
        destination = Vector3.zero;
    }

    public Vector3 GetDestination()
    {
        return destination;
    }

    public bool HasDestination()
    {
        return has_destination;
    }

    private Vector3 rotate(Vector3 oD,float theta, bool clockwise)
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

    public bool reachedDestination()
    {
        // Debug.Log("Distance: " + Vector3.Distance(body.position, destination));
        return Vector3.Distance(body.position, destination) < 0.76f;
    }



}