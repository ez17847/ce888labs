#!/usr/bin/env python
from sopel import module
from emo.wdemotions import EmotionDetector

emo = EmotionDetector()

@module.rule('')

def hi(bot, trigger):
	a = 1
	print(trigger, trigger.nick)
	#bot.say(trigger, trigger.nick)
	raw = emo.detect_emotion_in_raw_np(trigger)
	print(raw)
	emotion0 = int(raw[0])
	emotion1 = int(raw[1])
	emotion2 = int(raw[2])
	emotion3 = int(raw[3])
	emotion4 = int(raw[4])
	emotion5 = int(raw[5])
	emotion = '{} {} {} {} {} {} {} {}'.format('[',emotion0,emotion1,emotion2,emotion3,emotion4,emotion5,']')
	#trigger.nick = emotion
	bot.say(emotion)
	if raw[0] > 0:
		bot.say("The emotion is Anger")
	if raw[1] > 0:
		bot.say("The emotion is Disgust")
	if raw[2] > 0:
		bot.say("The emotion is Fear")
	if raw[3] > 0:
		bot.say("The emotion is Joy")
	if raw[4] > 0:
		bot.say("The emotion is Sadness")
	if raw[5] > 0:
		bot.say("The emotion is Surprise")

