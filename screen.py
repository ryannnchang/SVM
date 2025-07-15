import qwiic_oled
import sys
import time
import RPi.GPIO as GPIO

#Defining 
myOLED = qwiic_oled.QwiicMicroOled()

#Setting Up
myOLED.begin()
myOLED.clear(myOLED.ALL)
myOLED.clear(myOLED.PAGE)

#Monitor dimensions
w = myOLED.get_lcd_width()
h = myOLED.get_lcd_height()

#Display Function
def display(word):
	myOLED.clear(myOLED.PAGE)
	myOLED.print(word)
	myOLED.set_cursor(0,0)
	myOLED.pixel(32, 32)
	myOLED.display()
	

def off():
	myOLED.clear(myOLED.ALL)

