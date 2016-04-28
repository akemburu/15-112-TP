import csv 
import pickle


mydict = {"akemburu": [(1,5,"sleepy"),(2,6,"awake")], "batman": [(2,4,"awake")], "jp": [(2,5, "sleepy")]}

#[(string, [(int, int, string),(int,int,string)),(string, [(int, int, string),(int,int,string)),(string, [(int, int, string),(int,int,string))]

def writeUsertoCSV(userDict): 
	newDict = pickle.dump( userDict, open( "save.p", "wb" ) )

def userDictionary(): 
	favorite_color = pickle.load( open( "save.p", "rb" ) )
	return favorite_color


def dictionaryModification(date, sleepHours, username, state): 
	userDict = userDictionary()
	print(userDict)
	if username in userDict: 
	    currentValue = userDict[username]
	    currentValue.append((date, sleepHours,state))
	    userDict[username] = currentValue
	else: 
		userDict[username] = []
		currentValue = userDict[username]
	   	currentValue.append((date,sleepHours,state))
	writeUsertoCSV(userDict)
	return userDict

#writeUsertoCSV(mydict)
#dictionaryModification(11, 10, "ronag", "awake")
print(userDictionary())