import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99         
TAU = 1e-2           
LR_ACTOR = 1e-3      
LR_CRITIC = 1e-2     
WEIGHT_DECAY = 0     
NUM_AGENTS = 2
STATE_SIZE = 24 
ACTION_SIZE = 2 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiagentWrapper:
    """The two agents are entangled with a shared replay buffer and critic networks that take both of their
    observation and action spaces as inputs. This is orchestrated by this class to declutter the IPython Notebook."""
    
    def __init__(self, random_seed, batch_size, buffer):
        self.agents = [ Agent(agentNr, STATE_SIZE, ACTION_SIZE, random_seed) for agentNr in range(0, NUM_AGENTS)]
        random.seed(random_seed)
        self.batch_size = batch_size
        np.random.seed(random_seed)
        self.buffer = buffer
        
        
    def learn(self):
        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            states, actions, rewards, next_states, dones = experiences
            
            next_actions = torch.cat([
                self.agents[agentNr].actor_target(next_states[:, (STATE_SIZE*agentNr):(STATE_SIZE*(agentNr + 1))])
                for agentNr in range(0, NUM_AGENTS)], dim=1)
            
            for agentNr in range(0, NUM_AGENTS):
                self.agents[agentNr].learn(states[:, (STATE_SIZE*agentNr):(STATE_SIZE*(agentNr + 1))],
                      actions[:, (ACTION_SIZE*agentNr):(ACTION_SIZE*(agentNr + 1))],
                      rewards[:, agentNr],
                      next_states[:, (STATE_SIZE*agentNr):(STATE_SIZE*(agentNr + 1))],
                      dones[:, agentNr],
                      states,
                      next_states,
                      actions,
                      next_actions, GAMMA)
    
    def act(self, states, noise_level=1.0):
        return np.vstack([ self.agents[agentNr].act(states[agentNr, :][None, :], noise_level) for agentNr in range(0, NUM_AGENTS)])

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_nr, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            ageent_nr (int): number of agent, either 0 or one
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.agent_nr = agent_nr
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device) 
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)

        # Actor Network (w/ Target Network)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


        self.soft_update(self.critic_local, self.critic_target, 1) # should amount to a hard update
        self.soft_update(self.actor_local, self.actor_target, 1)   # should amount to a hard update 
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, noise_level):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += noise_level * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, states, actions, rewards, next_states, dones, all_states, all_next_states, all_actions, all_next_actions, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #states, actions, criticinputs, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #actions_next = self.actor_target(next_states)
        self.critic_optimizer.zero_grad()
        
        with torch.no_grad():
            Q_targets_next = self.critic_target(all_next_states, all_next_actions).squeeze() 
        
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.critic_local(all_states, all_actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets.unsqueeze(1))
        
        critic_loss.backward()

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.actor_optimizer.zero_grad()
        if self.agent_nr == 0:
            actions_pred = torch.cat((self.actor_local(states), all_actions[:, self.action_size:(2*self.action_size)]), dim = 1)
        else:
            actions_pred = torch.cat((all_actions[:, 0:self.action_size], self.actor_local(states)), dim = 1)

        actor_loss = -self.critic_local(all_states, actions_pred).mean()
 
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([np.hstack(e.states) for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([np.hstack(e.actions) for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([np.hstack(e.next_states) for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)