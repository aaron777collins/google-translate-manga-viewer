import subprocess
import sys
import pyautogui
def install(package):
	subprocess.check_call([sys.executable, "-m", "pip", "install", package])


#Ensures pip is installed:
subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])


install('pyautogui')
install('opencv-python')

worked = False
confidence_amount = 1.0
while(worked != True):
	try:
		x, y = pyautogui.locateCenterOnScreen('images\\translateimages0.png', confidence=confidence_amount)
		pyautogui.moveTo( (x, y), duration=0.25)
		pyautogui.click((x, y))
		worked = True
	except TypeError:
		confidence_amount-=0.1
pyautogui.moveTo((2699, 545), duration=2.0)
worked = False
confidence_amount = 1.0
while(worked != True):
	try:
		x, y = pyautogui.locateCenterOnScreen('images\\translateimages1.png', confidence=confidence_amount)
		pyautogui.moveTo( (x, y), duration=0.25)
		pyautogui.click((x, y))
		worked = True
	except TypeError:
		confidence_amount-=0.1
worked = False
confidence_amount = 1.0
while(worked != True):
	try:
		x, y = pyautogui.locateCenterOnScreen('images\\translateimages2.png', confidence=confidence_amount)
		pyautogui.moveTo( (x, y), duration=0.25)
		pyautogui.click((x, y))
		worked = True
	except TypeError:
		confidence_amount-=0.1
