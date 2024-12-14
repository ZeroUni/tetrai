## TetrAI
An attempt to make a visual learning tetris AI

Current Plan
```mermaid
graph TD;
    A[Set up PyTorch-based CNN] -->|Input: Image| B[CNN outputs a tokenized move];
    C[Set up a Tetris environment] -->|Move: Token| D[Environment outputs a new state as an image];
    B -->|Training| E[Train the CNN with custom rewards and penalties];
    D -->|State: Image| E;
    E -->|Iteration| F[Repeat until the CNN can play the game];
    F -->|Success| G[Profit];
    
    style A fill:#a2d5f2,stroke:#333,stroke-width:2px;
    style B fill:#a8e6cf,stroke:#333,stroke-width:2px;
    style C fill:#ffd3b6,stroke:#333,stroke-width:2px;
    style D fill:#d7aefb,stroke:#333,stroke-width:2px;
    style E fill:#fff7a3,stroke:#333,stroke-width:2px;
    style F fill:#ffb7ce,stroke:#333,stroke-width:2px;
    style G fill:#cfd8dc,stroke:#333,stroke-width:2px;

    classDef textColor fill:#000,stroke:#333,stroke-width:3px,color:#000;
    class A,B,C,D,E,F,G textColor;
```

## Commands
```bash
python tetrai/RewardWeights.py
```
- Example run
```bash
python tetrai/DDQNonCNN.py --num_episodes 2000 --max_moves 200 --weights 'E:\CodingProjects\tetrai\tetrai\weights.json'
```
```bash
python tetrai/PPOonCNN.py --num_episodes 2000 --max_moves 200 --weights 'E:\CodingProjects\tetrai\tetrai\weights.json'
```

- Prev run
```bash
python tetrai/DDQNonCNN.py --num_episodes 10000 --max_moves 40 --weights 'E:\CodingProjects\tetrai\tetrai\reward_only.json' --save_interval 250 --cycles 16 --learning_rate .0001 --resume True --policy_net 'E:\CodingProjects\tetrai\out\1734167643\tetris_policy_net_500.pth' --target_net 'E:\CodingProjects\tetrai\out\1734167643\tetris_target_net_500.pth' --level_inc 3000
```