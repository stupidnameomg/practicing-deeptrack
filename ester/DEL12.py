# UPPGIFT 1.2.1
txt = input(f'Skriv några tal separerade med mellanslag: ')
numbers = txt.split(' ')

# Gör om till int för att kunna använda min och max (för stringar kommer de jämföra siffror ist)
for i in range(0,len(numbers)):
    numbers[i] = int(numbers[i])

# HÖGSTA TAL
max = max(numbers)
print (f'Det största talet är:  '+ str(max))

# LÄGSTA TAL
min = min(numbers)
print (f'Det minsta talet är: ' + str(min))

# MEDELVÄRDE
sum = 0
for i in range(0, len(numbers)):
    sum+= numbers[i]

medel = sum / len(numbers)
print (f'Medelvärdet av talen är: ' + str(medel))

###############################################################
# UPPGIFT 1.2.2

mixt = input(f'Skriv något med siffror och bokstäver: ')

digits = 0
letters = 0

# Räknar antalet bokstäver och siffror
for c in mixt:
    if c.isdigit() == True: # isdigit ger True omm c är en siffra
        digits +=1
    elif c.isalpha() == True: # isalpha ger True omm c är en bokstav
        letters +=1

print(f'Antal siffror:' + str(digits) + ', Antal bokstäver: ' + str(letters))