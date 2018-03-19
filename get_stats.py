from sys import argv
from subprocess import check_output
import operator
out = check_output('grep -n DONE ' + argv[1], shell=True)
out = out.split('\n')
print out.count("unning")
#print out
start = out[-3].split(':')[0]
end = out[-2].split(':')[0]
print "Start:", start
print "End:", end
all_last = check_output('head -n' + end + " " + argv[1] + " | /usr/bin/tail -n" + str(int(end)-int(start)), shell=True)
#print all_last
#print all_last.count("unning")
#print check_output("head -n 140263 " + argv[1]  + " | tail -n" + str(140263 - 139908), shell=True)
#print check_output("/usr/bin/which head", shell=True)
assert all_last.count("DONE") == 1
print "Count:", all_last.count("Nodes")
print "Top 20:"
count = 0
linesdict = dict() 
for line in all_last.split('\n'):
    val = line.split(" ")[-1]
    if "Nodes" in line and val != "nan":
        linesdict[line] = float(val)

sorteddict = sorted(linesdict.items(), key=operator.itemgetter(1))
for result in sorteddict:
    line = result[0]
    count += 1
    print line
    if count >= 20:
        break
