import re
from vap_datasets.utils.utils import read_txt
import datetime
import json

"""
Transcript

&Tufts
%um/uh: non-lexemes
{laugh}
{lipsmack}

see


Special symbols:
    Noises, conversational phenomena, foreign words, etc. are marked
    with special symbols.  In the table below, "text" represents any
    word or descriptive phrase.

    {text}              sound made by the talker {laugh} {cough} {sneeze} {breath}
    [text]              sound not made by the talker (background or channel)
                        [distortion]    [background noise]      [buzz]

    [/text]             end of continuous or intermittent sound not made by
                        the talker (beginning marked with previous [text])

    [[text]]            comment; most often used to describe unusual
                        characteristics of immediately preceding or following
                        speech (as opposed to separate noise event)

                        [[previous word lengthened]]    [[speaker is singing]]

    ((text))            unintelligible; text is best guess at transcription
                        ((coffee klatch))

    (( ))               unintelligible; can't even guess text
                        (( ))


    <language text>     speech in another language
                        <English going to California>

    <? (( ))>           ? indicates unrecognized language; 
                        (( )) indicates untranscribable speech
                        <? ayo canoli>  <? (( ))>

    -text		partial word
    text-               -tion absolu- 

    #text#              simultaneous speech on the same channel
                        (simultaneous speech on different channels is not
                        explicitly marked, but is identifiable as such by
                        reference to time marks)

    //text//            aside (talker addressing someone in background)
                        //quit it, I'm talking to your sister!//

    +text+              mispronounced word (spell it in usual orthography) +probably+

   **text**             idiosyncratic word, not in common use
                        **poodle-ish**

    %text               This symbol flags non-lexemes, which are
			general hesitation sounds.  See the section on
			non-lexemes below to see a complete list for
			each language.  
			%mm %uh 

    &text               used to mark proper names and place names
                        &Mary &Jones    &Arizona        &Harper's
                        &Fiat           &Joe's &Grill

    text --             marks end of interrupted turn and continuation
    -- text             of same turn after interruption, e.g.
                        A: I saw &Joe yesterday coming out of --
                        B: You saw &Joe?!
                        A: -- the music store on &Seventeenth and &Chestnut.
"""


def csj_regexp(s):
    """travelagency specific regexp"""
    # {text}:  sound made by the talker {laugh} {cough} {sneeze} {breath}
    s = re.sub(r"\{.*\}", "", s)

    # [[text]]: comment; most often used to describe unusual
    s = re.sub(r"\[\[.*\]\]", "", s)

    # [text]: sound not made by the talker (background or channel) [distortion]    [background noise]      [buzz]
    s = re.sub(r"\[.*\]", "", s)

    # (( )): unintelligible; can't even guess text
    s = re.sub(r"\(\(\s*\)\)", "", s)

    # single word ((taiwa))
    # ((text)): unintelligible; text is best guess at transcription ((coffee klatch))
    s = re.sub(r"\(\((\w*)\)\)", r"\1", s)

    # multi word ((taiwa and other))
    # s = re.sub(r"\(\((\w*)\)\)", r"\1", s)
    s = re.sub(r"\(\(", "", s)
    s = re.sub(r"\)\)", "", s)

    # -text / text-: partial word = "-tion absolu-"
    s = re.sub(r"\-(\w+)", r"\1", s)
    s = re.sub(r"(\w+)\-", r"\1", s)

    # +text+: mispronounced word (spell it in usual orthography) +probably+
    s = re.sub(r"\+(\w*)\+", r"\1", s)

    # **text**: idiosyncratic word, not in common use
    s = re.sub(r"\*\*(\w*)\*\*", r"\1", s)

    # remove proper names symbol
    s = re.sub(r"\&(\w+)", r"\1", s)

    # remove non-lexemes symbol
    s = re.sub(r"\%(\w+)", r"\1", s)

    # text --             marks end of interrupted turn and continuation
    # -- text             of same turn after interruption, e.g.
    s = re.sub(r"\-\-", "", s)

    # <language text>: speech in another language
    s = re.sub(r"\<\w*\s(\w*\s*\w*)\>", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    s = re.sub(r"^\s", "", s)
    return s


def preprocess_utterance(filepath):
    """
    Load filepath and preprocess the annotations

    * Omit empty rows
    * join utterances spanning multiple lines
    """
    f = open(filepath, "r")
    data = json.load(f)
    return data


def load_utterances(filepath, clean=True):
    #try:
    data = preprocess_utterance(filepath)
    # except:
    #     print(f"ERROR on preprocess {filepath}")
    
    utterances = []

    #try:
    for row in data:
        start = time_to_seconds(row["starttime"])
        end = time_to_seconds(row["endtime"])
        speaker = 0 if row["speaker"] == "operator" else 1
        text = row["utterance"]
        if clean:
            text = csj_regexp(text)


        utterances.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )
        #print(start,end,speaker)
    # except:
    #     print(f"travelagency UTILS load_utterance")
    #     print(f"ERROR on split {filepath}")

    return utterances

def time_to_seconds(time):
    "03:14.70"
    mins, secs = time.split(":")
    mins = int(mins)
    secs = float(secs)
    return mins*60 + secs


def extract_vad(path):
    utterances = load_utterances(path)
    vad = [[], []]
    for utt in utterances:
        vad[utt["speaker"]].append((utt["start"], utt["end"]))
    return vad



if __name__ == "__main__":

    filepath = "/data/group1/z40351r/datasets_turntaking/data/CSJ/TRN/Form2/noncore/D03M0013.trn"
    vad = extract_vad(filepath)
    print(vad)
    