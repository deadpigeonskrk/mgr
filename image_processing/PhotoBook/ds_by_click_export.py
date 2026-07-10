# %load_ext autoreload
# %autoreload 2
from logsProcessing_export import display_round_stripe, round_list_tpl, Log

def create_click_DS(input_ds):
    counter = 5
    dict_clic_tpls = {}
    for round_tuple in input_ds:
        if counter > 0:
            round_data = round_tuple[0]
            game_id = round_tuple[1]

            round_name = "{}_{}".format(game_id, round_data.round_nr)
            
            dial_string = ""
            click_counter = 0
            
            for message in round_data.messages:
                if message.type == "text":
                    dial_string += "{}: {}\n".format(message.speaker, message.text)
        
                if message.type == "selection":
                    click_counter += 1
                    label = "common" if message.text.split()[1] == "<com>" else "different"
                    marking_act = "{} marks image {} as {}\n".format(message.speaker, Log.strip_image_id(message.text.split()[2]), label)

                    dict_clic_tpls[round_name + "_" + str(click_counter)] = (dial_string, marking_act, message.speaker, Log.strip_image_id(message.text.split()[2]), label, round_data.images)
                    dial_string += marking_act

                                            
            counter -= 1
    return dict_clic_tpls
click_DS = create_click_DS(round_list_tpl)

