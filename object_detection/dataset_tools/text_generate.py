def generate_number(i):
    if i>=0 and i<10:
        return '0000'+str(i)
    elif i>=10 and i<100:
        return '000'+str(i)
    elif i>=100 and i<1000:
        return '00' + str(i)
    elif i >= 1000 and i < 10000:
        return '0' + str(i)
    else:
        return str(i)

content=""
total=43971
for i in range(total):
    ii=generate_number(i)
    if i == total-1:
        content = content + ii
    else:
        content=content+ii+'\n'


f = open("train.txt",'w')
f.write(content)
f.close()