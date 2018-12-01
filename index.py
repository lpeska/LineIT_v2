# -*- coding: utf-8 -*-
import urllib,urllib2
import cgi
import cgitb; cgitb.enable() # Optional; for debugging only
print ("HTTP/1.0 200 OK\n")
f = cgi.FieldStorage()
query = ""
#for i,v in f.iteritems():
dct = {}
query = []
for i in f.keys():
        #query = query + i + "=" + f[i].value + "&"
        if type(f[i]) == list:
          dct[i] = ",".join(list(set([x.value for x in f[i]])))        
        else:
          dct[i] = f[i].value

        
dct["uid"] = 1
query = urllib.urlencode(dct, True) 

#print "http://127.0.0.1:50000/?"+query[:-1]

page = urllib2.urlopen("http://127.0.0.1:50001/?"+query)


#print "Content-Type: text/html\n\n\n"
print(page.read())

#print(f)
#print(dct)
#print(query)