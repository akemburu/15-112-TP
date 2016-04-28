import pickle

mydict = {"akemburu": [(1,5,"sleepy"),(2,6,"awake")], "batman": [(2,4,"awake")], "jp": [(2,5, "sleepy")]}

pickle.dump( mydict, open( "save.p", "wb" ) )

favorite_color = pickle.load( open( "save.p", "rb" ) )

print(type(favorite_color))