import json
import time
import argparse


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='json file')
	parser.add_argument('json', type=str, help='dir of json')

	args = parser.parse_args()


	with open(args.json) as f:
		data = json.load(f)


	c_total = 0
	c_25 = 0
	c_m100 = 0 # minus 100
	for gameid in data.keys():
		if data[gameid]['reward_player1'] == 25:
			c_25 += 1
		elif data[gameid]['reward_player1'] == -100:
			c_m100 += 1

		c_total += 1


		if int(gameid) % 5000 ==0:
			print (gameid, 100*c_25/c_total)
