def generate_prompt1(question="what is in the background of the video?"):
    """
    Generate a prompt for the main query based on the question and plans.
    """
    prompt1_format = f"""
Answer the main query based on 'question', 'plans' below.
Main query: Complete the following 'plans' to answer the 'question'.
Plans:
1. trim("choose one of the followings: beginning, middle, end, none", "truncated_question"):response
2. parse_event("choose one of the followings: before, when, after, none", "event_to_localize", "truncated_question"):response
3. classify("choose one of the followings: what, where, counting, why, how, location, when, who"):response
4. require_ocr("choose one of the followings: yes, no"):response

Question: how many children are in the video?
Answer:
trim("none"):no temporal hint found
parse_event("none"):no temporal conjunction found
classify("counting"):question asks the number of children
require_ocr("no"):question does not require reading texts

Question: how many animals are involved in the video?
Answer:
trim("none"):no temporal hint found
parse_event("none"):no temporal conjunction found
classify("counting"):question asks the number of animals
require_ocr("no"):question does not require reading texts

Question: where is this video taken?
Answer:
trim("none"):no temporal hint found
parse_event("none"):no temporal conjunction found
classify("location"):question asks the location of the place
require_ocr("no"):question does not require reading texts

Question: how is the woman feeling while playing with the girl in grey?
Answer:
trim("none"):no temporal hint found
parse_event("when", "woman playing with the girl in grey", "how is the woman feeling?"):'while' conjunction is found in the question
classify("how"):question asks the feeling of the woman
require_ocr("no"):question does not require reading texts

Question: where is this place?
Answer:
trim("none"):no temporal hint found
parse_event("none"):no temporal conjunction found
classify("location"):question asks the location of the place
require_ocr("no"):question does not require reading texts

Question: what does the baby do after finishing playing with the pram near the end?
Answer:
trim("end", "what does the baby do after finishing playing with the pram?"):'near the end' is found from the question
parse_event("after", "baby finishing playing with the pram", "what is the baby doing?"):'after' conjunction is found in the question
classify("what"):question asks what the baby is doing
require_ocr("no"):question does not require reading texts

Question: how do the boys in blue react after the two girls with red hair leave the stage at the end of the
video?
Answer:
trim("end", "how do the boys in blue react after the two girls with red hair leave the stage?"):'at the end of the video' is found from the question
parse_event("after", "girl with red hair leaving the stage", "how are the boys in blue reacting?"):'after' conjunction is found in the question
classify("what"):question asks how the boys react, meaning what the boys are doing
require_ocr("no"):question does not require reading texts

Question: what is this video about?
Answer:
trim("none"):no temporal hint found
parse_event("none"):no temporal conjunction found
classify("what"):question asks what the video is about
require_ocr("no"):question does not require reading texts

Question: what colour are all of these dogs?
Answer:
trim("none"):no temporal hint found
parse_event("none"):no temporal conjunction found
classify("what"):question asks what the colour of the dogs is
require_ocr("no"):question does not require reading texts

Question: how does the woman react as the tallest man start filming her?
Answer:
trim("none"):no temporal hint found
parse_event("when", "man start filming the woman", "how is the woman reacting?"):'as' conjunction is found in the question
classify("what"):question asks how the woman reacts, meaning what the woman is doing
require_ocr("no"):question does not require reading texts

Question: what did the man do before he kept the guitar in the red box at the start?
Answer:
trim("beginning", "what did the man do before he kept the guitar in the red box?"):'at the start' is found from the question
parse_event("before", "man keeping the guitar in the red box", "what is the man doing?"):'before' conjunction is found in the question
classify("what"):question asks what the man is doing
require_ocr("no"):question does not require reading texts

Question: what did the brown dog do right before standing up near the middle of the video?
Answer:
trim("middle", "what did the brown dog do before standing up?"):'middle of the video' is found from the question
parse_event("before", "brown dog standing up", "what is the brown dog doing?"):'right before' conjunction is found in the question
classify("what"):question asks what the brown dog is doing
require_ocr("no"):question does not require reading texts

Question: where are two men in blue at?
Answer:
trim("none"):no temporal hint found
parse_event("none"):no temporal conjunction found
classify("where"):question asks the location of the man
require_ocr("no"):question does not require reading texts

Question: how did the laptop fall after being turned off and closed?
Answer:
trim("none"):no temporal hint found
parse_event("after", "laptop turning off and closed", "how is the laptop falling?"):'after' conjunction is found in the question
classify("how"):question asks how the laptop falls
require_ocr("no"):question does not require reading texts

Question: what type of sports are three boys in striped playing at the start of the video?
Answer:
trim("beginning", "what type of sports are three boys in striped playing?"):'at the start of the video' is found from the question
parse_event("none"):no temporal conjunction found
classify("what"):question asks the type of sports that boys are playing
require_ocr("no"):question does not require reading texts

Question: {question}
Answer:
"""
    return prompt1_format.format(question=question)


def get_calls_prompt1(output, question):
    """
    Trim the output of generate_prompt1 based on the question.
    """
    if not output or not question in output:
        return []
    trimmed_list = output.split("Answer:\n")[-1].split("\n")
    calls = [line.split(":")[0].strip() for line in trimmed_list[:-1]]
    return calls


def generate_prompt2(phrase="what did the man throw?"):
    """
    Generate a prompt for the main query based on the phrase.
    """
    prompt2_format = f"""
Generate a short program given input 'phrase'.
Phrase:
- baby running for a while
Answer:
baby = localize(noun="baby", noun_with_modifier="")
verify_action("is the baby running?", [baby])

Phrase:
- how many children are in the video?
Answer:
children = localize(noun="child", noun_with_modifier="")
verify_action("no_action", [children])

Phrase:
- grey animals jumping up and down
Answer:
grey_animal = localize(noun="animal", noun_with_modifier="grey animal")
verify_action("is the animal jumping?", [grey_animal])

Phrase:
- how many animals are involved in the video?
Answer:
animals = localize(noun="animal", noun_with_modifier="")
verify_action("no_action", [animals])

Phrase:
- children in colorful t-shirts playing soccer in a playground
Answer:
children_in_colorful_tshirt = localize(noun="child", noun_with_modifier="child in colorful t-shirt")
playground = localize(noun="playground", noun_with_modifier="")
verify_action("is the children playing soccer in a playground?", [children_in_colorful_tshirt, playground])

Phrase:
- where is this video taken?
Answer:
no_localization = localize(noun="", noun_with_modifier="")
verify_action("no_action", [no_localization])

Phrase:
- what did the lions catch?
Answer:
lions = localize(noun="lion", noun_with_modifier="")
verify_action("is the lions catching something?", [lions])

Phrase:
- how did the group of people interact with the animals?
Answer:
people = localize(noun="person", noun_with_modifier="")
animals = localize(noun="animal", noun_with_modifier="")
verify_action("are the people interacting with animals?", [people, animals])

Phrase:
- men in red playing with the dogs
Answer:
men_in_red = localize(noun="man", noun_with_modifier="man in red")
dogs = localize(noun="dog", noun_with_modifier="")
verify_action("are the men playing with the dogs?", [men_in_red, dogs])

Phrase:
- why is the man in black raise his hand?
Answer:
man_in_black = localize(noun="man", noun_with_modifier="man in black")
verify_action("is the man raising his hand?", [man_in_black])

Phrase:
- woman playing with the girl in grey
Answer:
woman = localize(noun="woman", noun_with_modifier="")
girl_in_grey = localize(noun="girl", noun_with_modifier="girl in grey")
verify_action("is the woman playing with the girl?", [woman, girl_in_grey])

Phrase:
- how are the boys in blue clapping their hands?
Answer:
boys_in_blue = localize(noun="boy", noun_with_modifier="boy in blue")
verify_action("are the boys clapping their hands?", [boys_in_blue])

Phrase:
- baby finishing playing with the pram
Answer:
baby = localize(noun="baby", noun_with_modifier="")
pram = localize(noun="pram", noun_with_modifier="")
verify_action("is the baby playing with the pram?", [baby, pram])

Phrase:
- what is this video about?
Answer:
no_localization = localize(noun="", noun_with_modifier="")
verify_action("no_action", [no_localization])

Phrase:
- What is on the man's hat?
Answer:
man = localize(noun="man", noun_with_modifier="")
hat = localize(noun="hat", noun_with_modifier="")
verify_action("no_action", [man, hat])

Phrase:
- where is the woman's robe?
Answer:
womans_robe = localize(noun="robe", noun_with_modifier="woman's robe")
verify_action("no_action", [womans_robe])

Phrase:
- man keeping the guitar in the red box
Answer:
man = localize(noun="man", noun_with_modifier="")
guitar = localize(noun="guitar", noun_with_modifier="")
red_box = localize(noun="box", noun_with_modifier="red box")
verify_action("is the man keeping the guitar in the box?", [man, guitar, red_box])

Phrase:
- why are the men with red hat dancing?
Answer:
men_with_red_hat = localize(noun="man", noun_with_modifier="man with red hat")
verify_action("are the men dancing?", [men_with_red_hat])

Phrase:
- women in yellow looking for something
Answer:
women_in_yellow = localize(noun="woman", noun_with_modifier="woman in yellow")
verify_action("is the women looking for something?", [women_in_yellow])

Phrase:
- how is the woman reacting?
Answer:
woman = localize(noun="woman", noun_with_modifier="")
verify_action("no_action", [woman])

Phrase:
- what type of sports are three boys in striped playing?
Answer:
boys_in_striped = localize(noun="boy", noun_with_modifier="boy in striped")
verify_action("are the boys playing sports?", [boys_in_striped])

Phrase:
- what is the baby holding?
Answer:
baby = localize(noun="baby", noun_with_modifier="")
verify_action("is the baby holding something?", [baby])

Phrase:
- white cap and blue cap man dancing towards each other
Answer:
man_with_white_cap = localize(noun="man", noun_with_modifier="man with white cap")
man_with_blue_cap = localize(noun="man", noun_with_modifier="man with blue cap")
verify_action("is the man with white cap and man white blue cap are dancing towards each other?", [man_with_white_cap, man_with_blue_cap])

Phrase:
- {phrase}
Answer:
"""
    return prompt2_format.format(phrase=phrase)


def get_calls_prompt2(output, phrase):
    """
    Trim the output of generate_prompt1 based on the question.
    """
    if not output or not phrase in output:
        return []
    trimmed_list = output.split("Answer:\n")[-1].split("\n")
    calls = [line.split("\n")[0].strip() for line in trimmed_list[:-1]]
    return calls


def generate_prompt3(qatype="what", question="what is the cat doing?"):
    """
    Generate a prompt for the main query based on the question type and question.
    """

    prompt3_format = f"""
Answer the main query based on 'question type', 'question', and 'plans'.
Main query: Complete the following 'plans' to answer the 'question'.
Plans:
1. vqa("question", require_ocr)
2. vqa(["list of at most three distinct questions that support answering the question; there can be no supporting questions based on 'question type'"], require_ocr):response

Question type: what
Question: What is the girl holding in her hands?
Answer:
vqa("What is the girl holding in her hands?", require_ocr=False)
vqa([], require_ocr=False):question type 'what' is given which is sufficiently simple to answer, thus empty supporting question list

Question type: why
Question: why does the boy in striped turn around?
Answer:
vqa("why does the boy in striped turn around?", require_ocr=False)
vqa(["what is the boy in striped doing?", "what surrounds the boy?"], require_ocr=False):question type 'why' is given which requires to ask context of the scene that the boy is in

Question type: what
Question: what is the lady doing?
Answer:
vqa("what is the lady doing?", require_ocr=False)
vqa([], require_ocr=False):question type 'what' is given which is sufficiently simple to answer, thus empty supporting question list

Question type: how
Question: how did the boy open the laptop?
Answer:
vqa("how is the boy opening the laptop?", require_ocr=False)
vqa(["what is the boy doing?", "what method is the boy using to open the laptop?", "what is the boy doing with the laptop?"], require_ocr=False):question type 'how' is given which requires to identify the context of the scene and the method that the boy is using to open the laptop

Question type: why
Question: why is the man holding?
Answer:
vqa("why is the man holding?", require_ocr=False)
vqa(["what is the man doing?", "what surrounds the man?", "what is the man holding?"], require_ocr=False):question type 'why' is given which requires to ask context of the scene that the man is in

Question type: location
Question: where is this video taken?
Answer:
vqa("where is this video taken?", require_ocr=False)
vqa(["where is this place?", "what can be seen in this image?", "identify main objects in the scene."], require_ocr=False):question type 'location' is given which requires to analyze objects in the image

Question type: what
Question: what does the man playing the drums do?
Answer:
vqa("what does the man playing the drums do?", require_ocr=False)
vqa([], require_ocr=False):question type 'what' is given which is sufficiently simple to answer, thus empty supporting question list

Question type: location
Question: where is this happening?
Answer:
vqa("where is this happening?", require_ocr=False)
vqa(["where is this place?", "what can be seen in this image?", "identify main objects in the scene."], require_ocr=False):question type 'location' is given which requires to analyze objects in the image

Question type: why
Question: why did the baby lean on the sofa?
Answer:
vqa("why did the baby lean on the sofa?", require_ocr=False)
vqa(["what are the baby doing?", "what surrounds the baby?"], require_ocr=False):question type 'why' is given which requires to ask context of the scene that the baby is in

Question type: where
Question: where is the boy projecting his photos on?
Answer:
vqa("where is the boy projecting his photos on?", require_ocr=False)
vqa(["what surrounds the boy?", "what is the boy doing?", "how is the boy projecting his photos?"], require_ocr=False):question type 'where' is given which requires to ask the context of the scene, surroundings of the boy, and the method that the boy is using to project his photos

Question type: how
Question: how did the two women react?
Answer:
vqa("how did the two women react?", require_ocr=False)
vqa(["what are the women doing?", "what surrounds the women?", "how are the women feeling?"], require_ocr=False):question type 'how' is given which requires to identify the context of the scene and the way that the two women interact

Question type: how
Question: how is the girl on the right feeling?
Answer:
vqa("how is the girl on the right feeling?", require_ocr=False)
vqa(["what is the girl on the right doing?", "is the girl on the right feeling positively or negatively?", "what surrounds the girl?"], require_ocr=False):question type 'how' is given which requires to identify the context of the scene and feeling of the girl

Question type: counting
Question: how many people are there?
Answer:
vqa("how many people are there?", require_ocr=False)
vqa(["is there more than one person?", "count the number of people."], require_ocr=False):question type 'counting' is given which requires to count

Question type: counting
Question: how many birds are involved in this video?
Answer:
vqa("how many birds are involved in this video?", require_ocr=False)
vqa(["is there more than one bird?", "count the number of birds."], require_ocr=False):question type 'counting' is given which requires to count

Question type: what
Question: What did he or she wear in his finger?
Answer:
vqa("What did the person wear in his finger?", require_ocr=False)
vqa([], require_ocr=False):question type 'what' is given which is sufficiently simple to answer, thus empty supporting question list

Question type: {qatype}
Question: {question}
Answer:
"""
    return prompt3_format.format(qatype=qatype, question=question)


summary = """
[frame 0] caption: a man in a gray shirt with the word mercury on it
[frame 1] caption: a man wearing a gray shirt with the word champions on it
[frame 2] caption: a man wearing a shirt that says ' pacific coast ' on it
[frame 3] caption: a man is kneeling down and working on a boat engine
[frame 4] caption: a man in a gray shirt is working on an outboard motor
[frame 5] caption: a man kneeling down looking under a boat engine
[frame 11] caption: a man wearing a grey shirt that says ' uhf ' on it
[frame 12] caption: a man wearing a t-shirt that says north shore marine
[frame 0] what is in the background?: rocks
[frame 1] what is in the background?: rocks and bushes
[frame 2] what is in the background?: a man kneeling down in front of a rock wall
[frame 10] what is in the background?: rocks
[frame 11] what is in the background?: a man kneeling down in front of a rock wall
[frame 12] what is in the background?: rocks
"""
question = "what is in the background?"


def generate_prompt5(summary=summary, question=question):
    prompt5_format = f"""
Answer the main query based on the 'video summary' and 'question' below.
Video summary:
{summary}
Question: {question}
Main query: Provide an answer to the Question. Keep your answer short and concise; your answer must be one or two words without any additional texts, characters, or spaces.
Answer:
"""
    return prompt5_format.format(summary=summary, question=question)
