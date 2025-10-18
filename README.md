# POLICY EVALUATION

## AIM
To evaluate and compare different policies in the Frozen Lake environment and find the best policy for reaching the goal successfully.

## PROBLEM STATEMENT
In the Frozen Lake environment, an agent must navigate from the start to the goal while avoiding holes. Movements are uncertain due to slipperiness. A policy guides the agent’s actions, but not all policies are effective. The task is to:

Evaluate a given policy (V1) using policy evaluation. Create and test a new policy (V2) to improve performance. Compare both policies based on success rate and rewards. Find the best policy for safely reaching the goal. This helps in identifying the most efficient way to complete the task.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by the policy at state s
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

## OUTPUT:
### POLICY 1:
<img width="534" height="147" alt="Screenshot 2025-10-18 111633" src="https://github.com/user-attachments/assets/da60c894-bb72-4e93-9933-1a49565464ab" />
<img width="658" height="166" alt="Screenshot 2025-10-18 111529" src="https://github.com/user-attachments/assets/7dd85ed7-e74c-45ac-bbe6-a2ebdfa85eba" />



### POLICY 2:

<img width="534" height="147" alt="Screenshot 2025-10-18 111633" src="https://github.com/user-attachments/assets/da60c894-bb72-4e93-9933-1a49565464ab" />
<img width="534" height="171" alt="Screenshot 2025-10-18 111542" src="https://github.com/user-attachments/assets/16272bf7-4f9d-4f40-a593-a309ed78f8f3" />


### COMPARISON:
<img width="485" height="175" alt="Screenshot 2025-10-18 111551" src="https://github.com/user-attachments/assets/0cd526e7-12cc-4975-9acd-d922698077a9" />

## RESULT:

Thus, The Python program to evaluate the given policy is successfully executed.
