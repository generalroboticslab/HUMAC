# HUMAC: Enabling Multi-Robot Collaboration from Single-Human Guidance
[Zhengran Ji](https://jzr01.github.io/)¹, [Lingyu Zhang](https://lingyu98.github.io/)¹, [Paul Sajda](https://liinc.bme.columbia.edu/people/paul-sajda)², [Boyuan Chen](http://boyuanchen.com/)¹

¹ Duke University, ² Columbia University

![Multi-Agent/Robot Collaboration](images/Teaser.jpeg)


## Overview
Learning collaborative behaviors is essential for multi-agent systems. Traditionally, multi-agent reinforcement learning solves this implicitly through a joint reward and centralized observations, assuming collaborative behavior will emerge. Other studies propose to learn from demonstrations of a group of collaborative experts. Instead, we propose an efficient and explicit way of learning collaborative behaviors in multi-agent systems by leveraging expertise from only a single human. Our insight is that humans can naturally take on various roles in a team. We show that agents can effectively learn to collaborate by allowing a human operator to dynamically switch between controlling agents for a short period and incorporating a human-like theory-of-mind model of teammates. Our experiments showed that our method improves the success rate of a challenging collaborative hide-and-seek task by up to 58% with only 40 minutes of human guidance. We further demonstrate our findings transfer to the real world by conducting multi-robot experiments.

![Method](images/Mainfig.jpeg)

## Result
## Simulation Success Rate (%)

| Setting                     | 2v1          | 2v2          | 3v2          | 3v3          | 4v3          | 4v4          |
|-----------------------------|--------------|--------------|--------------|--------------|--------------|--------------|
|                             |              |              |              |              |              |              |
| **Heuristic**               | 31.6±3.6     | 23.3±3.6     | 44.0±1.1     | 36.4±4.1     | 58.0±3.3     | 48.7±1.4     |
| **IL**                      | 17.1±3.1     | 7.1±1.4      | 16.4±1.3     | 12.0±2.4     | 23.8±3.5     | 19.1±3.0     |
| **IL-Long**                 | 71.8±3.1     | 55.3±2.0     | 77.6±1.7     | 66.4±0.8     | 85.1±1.7     | 81.3±3.8     |
| **IL FT**                   | 18.9±2.3     | 7.0±2.1      | 28.9±1.7     | 19.3±0.9     | 38.7±2.8     | 14.4±1.7     |
| **IL-Long FT**              | 64.7±0.5     | 46.2±3.5     | 74.7±0.5     | 66.0±5.0     | 88.0±2.0     | 80.7±1.4     |
| **PE-N**                    | 59.6±3.6     | 46.4±1.6     | 75.6±0.8     | 51.3±2.4     | 88.4±0.6     | 73.6±2.1     |
| **PE-H**                    | 71.8±3.2     | 51.6±4.0     | 70.2±0.6     | 58.2±2.1     | 84.9±3.3     | 81.6±3.6     |
| **PE-T**                    | 78.7±1.9     | 67.3±2.9     | 90.9±1.4     | **86.0±4.3** | **94.7±2.7** | **94.2±1.4** |
| **Combination**             | 1+1          | 1+1          | 2+1 | 1+2    | 2+1 | 1+2    |
| **IL-Long+IL-Long FT**      | 71.8±0.8     | 53.1±5.1     | 83.1±2.3     | 86.0±1.4     | 74.9±0.8     | 74.7±2.0     |
| **IL-Long+PE-N**            | 74.4±1.3     | 50.0±4.3     | 83.3±2.0     | 86.2±0.6     | 76.2±2.5     | 77.3±2.4     |
| **IL-Long+PE-H**            | 84.2±0.8     | 71.3±0.5     | 87.8±2.5     | 85.6±4.2     | 77.3±5.0     | 76.0±1.9     |
| **IL-Long+PE-T**            | **89.3±2.0** | **72.2±3.3** | **91.3±2.4** | **94.9±1.4** | **83.1±1.7** | **86.2±0.6** |

## Real-World Experiment Success Rate (%)
<table>
    <thead>
        <tr>
            <th>Setting</th>
            <th>2v1</th>
            <th>2v2</th>
            <th colspan="2">3v2</th>
            <th colspan="2">3v3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Heuristic</td>
            <td>11/20</td>
            <td>10/20</td>
            <td colspan="2">13/20</td>
            <td colspan="2">11/20</td>
        </tr>
        <tr>
            <td>IL-Long</td>
            <td>10/20</td>
            <td>7/20</td>
            <td colspan="2"><15/20</td>
            <td colspan="2">9/20</td>
        </tr>
        <tr>
            <td>Combination</td>
            <td>1+1</td>
            <td>1+1</td>
            <td>2+1</td>
            <td>1+2</td>
            <td>2+1</td>
            <td>1+2</td>
            
        </tr>
        <tr>
            <td>IL-Long + PE-T</td>
            <td><strong>12/20</strong></td>
            <td><strong>11/20</strong></td>
            <td><strong>16/20</strong></td>
            <td><strong>17/20</strong></td>
            <td><strong>16/20</strong></td>
        </tr>
    </tbody>
</table>



## Prerequisites

1. Clone the repository:

    ```bash
    git clone https://github.com/generalroboticslab/HUMAC.git
    ```
2. Install [CREW](https://github.com/generalroboticslab/CREW) for simulation

3. Install dependency for real-robot

   ```bash
   pip install -r requirements.txt
   ```
    
## Collecting Data

Run the following command to collect heuristic data or human involve data.

### simulation
```bash
cd Simulation/crew-algorithms/crew_algorithms/
export PYTHONPATH=..:$PYTHONPATH
WANDB_MODE=disabled python collect_data envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games]
```

### Real-world
```bash
cd Real-World/Hide-and-Seek-real-robot/
python collect_heuristic_data.py
```

## Training
Run the following command to train or fine-tune the model.

### simulation
```bash
cd Simulation/training/
python train.py
```

### real-world
```bash
cd Real-World/training/
python train.py 
```

## Evaluation

To evaluate the trained models and log the performance, run:

### simulation
```bash
cd Simulation/crew-algorithms/crew_algorithms/
export PYTHONPATH=..:$PYTHONPATH
WANDB_MODE=disabled python test envs.num_seekers=[num_seekers] envs.num_hiders=[num_hiders] envs.start_seed=[starting_seed] envs.num_games=[num_games]
```

### real-world

```bash
cd Real-World/Hide-and-Seek-real-robot/
python collect_heuristic_data.py
```

## Acknowledgement
[DJI Robomaster_sdk](https://github.com/dji-sdk/RoboMaster-SDK)
