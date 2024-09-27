using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Move : MonoBehaviour
{
    private float speed;
    public Rigidbody rb;
    Vector3 d;
    // Start is called before the first frame update
    void Start()
    {
        speed = 5f;
        gameObject.tag = "New"; 
        
    }

    // Update is called once per frame
    void FixedUpdate()
    {
       d = transform.forward;
       Vector3 Move = new Vector3(d.x*speed,0,d.z*speed);
       rb.velocity = Move;
       transform.forward  = Move;
    }

    void OnCollisionEnter(Collision collision)
    {

        if (collision.gameObject.CompareTag("Wall") || collision.gameObject.CompareTag("Obstacle"))
        {
            Vector3 wallNormal = collision.contacts[0].normal;
            Vector3 reflectedDirection = Vector3.Reflect(transform.forward, wallNormal);
            
            // Set the new direction for the player character
            transform.forward = reflectedDirection;
        }
        
    }
}

