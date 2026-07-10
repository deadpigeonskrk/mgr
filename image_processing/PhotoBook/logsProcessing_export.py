import os
import re
import datetime
from collections import defaultdict
import json
import re
import datetime
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import base64
import requests

class Log:
    def __init__(self, logfile):
        self.game_id = logfile['game_id']
        self.domain_id = logfile['domain_id']
        self.agent_ids = logfile['agent_ids']
        self.agent_labels = logfile['agent_labels']
        self.feedback = logfile['feedback']
        self.rounds, self.complete = self.load_rounds(logfile['rounds'], self.game_id)
        self.total_score = self.calculate_score()
        self.scores = self.calculate_player_scores()
        self.start_time = logfile['start_time']
        self.duration = self.calculate_duration()
        self.domains = self.get_domains()
        self.check_feedback()

    def load_rounds(self, game_rounds, game_id):
        rounds = []
        message_id = 0
        for round_data in game_rounds:
            game_round = GameRound(round_data, game_id, message_id)
            message_id = game_round.message_id
            rounds.append(game_round)
        if len(rounds) < 5:
            return (rounds, False)
        return (rounds, True)

    def calculate_score(self):
        total_score = 0
        for game_round in self.rounds:
            total_score += game_round.total_score
        return total_score

    def calculate_player_scores(self):
        player_scores = defaultdict(lambda: 0)
        for game_round in self.rounds:
            for player, score in game_round.scores.items():
                player_scores[player] += score
        return player_scores

    def get_domains(self):
        path = self.rounds[0].images["A"][0].split("/")[0]
        return [domain for domain in path.rsplit("_", 1)]

    def calculate_duration(self):
        start_time = self.rounds[0].messages[0].timestamp
        end_time = self.rounds[-1].messages[-1].timestamp
        return end_time - start_time

    def check_feedback(self):
        if "A" not in self.feedback:
            self.feedback["A"] = None
        if "B" not in self.feedback:
            self.feedback["B"] = None

    def format_time(datetime_obj):
        return datetime_obj.strftime('%M:%S')


    def strip_image_id(image_path):
        return int(image_path.split('_')[-1].split('.')[0].lstrip('0'))


class GameRound:
    def __init__(self, logfile_entry, game_id, message_id):
        self.round_nr = logfile_entry['round_nr'] + 1
        self.images = logfile_entry['images']
        self.common = logfile_entry['common']
        self.highlighted = logfile_entry['highlighted']
        self.scores = dict(logfile_entry['score'])
        self.total_score = self.calculate_score(logfile_entry['score'])
        self.messages, self.message_id = self.load_messages(logfile_entry['messages'], message_id)
        self.num_messages = self.count_text_messages()
        self.duration = self.calculate_duration()

    def load_messages(self, message_list, message_id):
        messages = []
        for message_data in message_list:
            message = Message(message_data, message_id)
            messages.append(message)
            message_id += 1

        if len(messages) == 0:
            print('ERROR: Missing messages for this game')
        return messages, message_id

    def calculate_score(self, score_dict):
        score = 0
        if not score_dict.values():
            return None
        for player_score in score_dict.values():
            score += player_score
        return score

    def count_text_messages(self):
        count = 0
        for message in self.messages:
            if message.type == "text":
                count += 1
        return count

    def calculate_duration(self):
        start_time = self.messages[0].timestamp
        for message in self.messages[::-1]:
            if message.type == 'feedback':
                end_time = message.timestamp
                return end_time - start_time


class Message:
    def __init__(self, logfile_message, message_id):
        self.message_id = message_id
        self.agent_id = logfile_message['agent_id']
        self.text = logfile_message['message']
        self.speaker = logfile_message['speaker']
        if message_id == 0:
            self.timestamp = datetime.datetime.strptime(logfile_message['timestamp'], '%H:%M:%S')
        else:
            self.timestamp = datetime.datetime.strptime(logfile_message['timestamp'], '%H:%M:%S.%f')
        self.turn = logfile_message['turn']
        self.type = self.determine_message_type()

    def determine_message_type(self):
        if not self.text.startswith("<"):
            return "text"
        else:
            return re.findall(r'<(.*?)>', self.text)[0]

def load_logs(log_repository, data_path):

    filepath = os.path.join(data_path, log_repository)
    print("Loading logs from {}...".format(filepath))

    missing_counter = 0
    file_count = 0
    for _, _, files in os.walk(filepath):
        file_count += len(files)
    print("{} files found.".format(file_count))
    logs = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), 'r') as logfile:
                    log = Log(json.load(logfile))
                    if log.complete:
                        logs.append(log)

    print("DONE. Loaded {} completed game logs.".format(len(logs)))
    return logs

data_path = ""
logs = load_logs("logs", data_path)

def create_round_dict(log_list):
    r_list = []
    for log in log_list:
        for round_data in log.rounds:
            # only rounds that are 100% correct
            if round_data.total_score == 6:
                r_list.append((round_data, log.game_id))
    return r_list

round_list_tpl = create_round_dict(logs)

def display_round_stripe(round_tuple, show=False):

    if show == True:
        round_data = round_tuple[0]
        game_id = round_tuple[1]
        round_name = "{}_{}".format(game_id, round_data.round_nr)
        
        image_A = Image.open(f"sets_images_stripe/{round_name}_A.jpg")
        image_B = Image.open(f"sets_images_stripe/{round_name}_B.jpg")
    
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
        axes[0].imshow(image_A)
        axes[0].set_title("A's view")
        axes[0].axis('off')
        axes[1].imshow(image_B)
        axes[1].set_title("B's view")
        axes[1].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    
        print("Round {}".format(round_name))
        print("\n")
    
        for message in round_data.messages:
            if message.type == "text":
                print("{}: {}".format(message.speaker, message.text))
    
            if message.type == "selection":
                label = "common" if message.text.split()[1] == "<com>" else "different"
                print("{} marks image {} as {}".format(message.speaker, Log.strip_image_id(message.text.split()[2]), label))

    return round_tuple[0].images