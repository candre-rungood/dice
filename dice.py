import time
import numpy as np
import ddqn_keras
import json

NUM_PLAYERS = 2 
NUM_DICES = 5
REWARD_WIN = 25
REWARD_LOSE = -100
REWARD_ACTION = -1
REWARD_INVALIDE = -200 # if IA take invalid action


class Player():
	def __init__(self, name='', q_table=False):
		dice = sorted([np.random.randint(1, 7) for i in range(1, 6)])
		self.dice = [np.sum([1 if d ==i else 0 for d in dice]) for i in range (1, 7)]
		self.game = None
		self.q_table = q_table
		self.name = name

	def add_game(self, game):
		self.game = game

	def action(self, action):
		d = action % 6 + 1
		n = int(np.floor(action/6)+3)
		print (f'{self.name} played {n} ge {d}')
		done = (11, 1) == (n, d)

		return (n, d), done, None


class Game():
	def __init__(self):
		self.previous_actions = [] # we can only start with (3,1)
		self.player0 = self.create_player('Player0') 
		self.player1 = self.create_player('Player1')
		self.players = [self.player0, self.player1]
		

	def counter_moves(self):
		return len(self.previous_actions)

	def create_player(self, name=''):
		player = Player(name)
		player.add_game(self)

		return player

	def reset(self):
		self.previous_actions = []
		self.player0 = self.create_player('Player0') 
		self.player1 = self.create_player('Player1')
		self.players = [self.player0, self.player1]
		print ('Player0', self.player0.dice)
		print ('Player1', self.player1.dice)

		return [self.player0.dice, self.player1.dice], self.actions()

	def step(self, action, counter_player):
		# which action and which player
		# returns observation_, reward, done, info 
		
		observation_, done, _ = self.players[counter_player].action(action)
		self.previous_actions.append(observation_)
		reward = self.get_reward(counter_player)
		#print ('self.players[counter_player].dice', self.players[counter_player].dice)
		observation_ = np.array(self.players[counter_player].dice+self.actions())

		return observation_, reward, done, None


	def is_finished(self):
		if len(self.previous_actions) == 0:
			return False

		return (11, 1) == self.previous_actions[-1]



	def actions(self, prev_action=None):
		# we want actions [(2, 1), (3, 1), (3, 2) .... (10, 5), (10, 6), (-1, -1)], len = 1+6*8+1
		# for now, cannot go back with lower n if we want to play 1s

		if prev_action == None:
			prev_action = self.previous_actions

	
		if len(prev_action) == 0:
			actions = [1 for i in range(6*8)]+[0] # cannot open first action
		else:
			#print ('prev', self.previous_actions)
			n, d = prev_action[-1]
			idx = (n-3)*6 + d  # index position 
			actions = [1 if i >= idx else 0 for i in range(6*8+1)]
		
		return actions


	def get_reward(self, counter_player):
		if not self.is_finished():
			reward = REWARD_ACTION
		else:
			n, d = self.previous_actions[-2] # last action before bluff was called.
			if d == 1:
				result = np.sum([player.dice[0] for player in self.players])
			else:
				result = np.sum([player.dice[d-1]+player.dice[0] for player in self.players])

			# which player won
			if counter_player==len(self.previous_actions)%2:
				#print ('c0', result)
				reward = REWARD_LOSE if result < n else REWARD_WIN
			else:
				reward = REWARD_WIN if result < n else REWARD_LOSE
				#print ('c1', result)

		return reward



if __name__ == '__main__':
    env = Game()

    ddqn_agent_player0 = ddqn_keras.DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=6*8+1, epsilon=1,
                  batch_size=64, input_dims=6*8+1+6, fname='models/player0')
    #ddqn_agent_player0.load_model(1000000)


    ddqn_agent_player1 = ddqn_keras.DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=6*8+1, epsilon=1,
                  batch_size=64, input_dims=6*8+1+6, fname='models/player1')
    #ddqn_agent_player1.load_model()
    
    ddqn_agent = [ddqn_agent_player0, ddqn_agent_player1]
    n_games = 100000
    ddqn_scores = []
    eps_history = []

    data_game = {}
    start_time = int(time.time())
    for i in range(n_games):
        done = False
        score = [0, 0]
        print ('')
        players_dice, observation = env.reset()
        #print (players_dice, observation)
        
        counter_player = 0 # keep track who is playing.
        counter_error = 0 # when model make an impossible action

        while not done:
            player_obs = np.array(players_dice[counter_player]+env.actions()) # we want each observation to be player' dice + actions
            action = ddqn_agent[counter_player].choose_action(player_obs)
 
            if env.actions()[action] == 1:            
                d = action % 6 + 1
                n = int(np.floor(action/6)+3)
                observation_, _, done, info = env.step(action, counter_player)
                observation = observation_ 
                counter_player += 1
                counter_player = counter_player % 2 
            else:
                # action non valide.
                print ('non valide')
                counter_error += 1
                reward = REWARD_INVALIDE
                done = False
                score[counter_player] += reward
                ddqn_agent[counter_player].remember(player_obs, action, reward, player_obs, int(done))

            
        data_game.update({i:{
				'timestamp': int(time.time()),
				'player1': env.players[0].name,
				'dice1': [int(x) for x in env.players[0].dice],
				'player2': env.players[1].name,
				'dice2': [int(x) for x in env.players[1].dice],
				'actions': [(int(n), int(d)) for (n, d) in env.previous_actions],
				'reward_player1': env.get_reward(0),
				'counter_error': counter_error
			}})
        #
        # 


        print ('-'*20)

        c = 0
        for n_ac, action in enumerate(env.previous_actions):

        	dice = env.players[c].dice
        	observation = np.array(dice+env.actions(env.previous_actions[:n_ac]))
        	if len(env.previous_actions)>n_ac+2:
        		observation_ = np.array(dice+env.actions(env.previous_actions[:n_ac+2]))
        	else:
        		observation_ = np.array(dice+env.actions(env.previous_actions))

        	
        	if env.previous_actions[n_ac] == (11, 1):
        		reward = env.get_reward(c)
        	elif env.previous_actions[n_ac+1] == (11,1):
        		reward = env.get_reward(c)
        	else:
        		reward = REWARD_ACTION

        	done = (n_ac+1) == len(env.previous_actions) or n_ac == len(env.previous_actions)
        	action = (action[0]-3)*6+action[1]-1
        	score[c] += reward
        	#print (observation, action, reward, observation_, int(done))
        	ddqn_agent[c].remember(observation, action, reward, observation_, int(done))
        	#print (i, observation, action, observation_, reward, c)
        	#print ('')

        	c += 1
        	c = c%2

        for c in range(2):
        	ddqn_agent[c].learn()
        #


        eps_history.append(ddqn_agent[0].epsilon)
        ddqn_scores.append(score[0])

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score[0],
              ' average score %.2f' % avg_score)

        if i>0:
	        log_i = np.log10(i)
	        if int(log_i) == log_i:
	            ddqn_agent[0].save_model(i)
	            ddqn_agent[1].save_model(i)

	        if i % 10000 == 0:
	            try:
	                with open(f'data_{start_time}.json') as f:
	                    data_ = json.load(f)
	                    data_.update(data_game)
	            except Exception as e:
	                    data_ = data_game

	            with open(f'data_{start_time}.json', 'w', encoding='utf-8') as f:
	                json.dump(data_, f, ensure_ascii=False, indent=4)
