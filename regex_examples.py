

import re

# For basics of regular expression syntax and semantics,
# read the documentation in re

str1 = "seed-1913232"
str2 = "seed-19237x"
str3 = "feed-19237"
digits_regex = "\d+" # a sequence of digits_regex of arbitrary length
regex1 = "seed-" + digits_regex # means: 'seed-' and then a sequence of digits_regex of arbitrary length



# return a match object if the regex matches the beginning of a string,
# None otherwise:

print (re.match(
                digits_regex,
                 str1
                 ))
# Output: None, because digits_regex does not match the beginning of str1

match =re.match(
    regex1,
    str1
)
print(match)

# Output: re.Match object which describes the match
# For example, this can be used as follows:

for string in (str1,str2,str3):
    if re.match(regex1,string):
        print (string)

# Let's say we want to get all matches of some regex in a string.
# For example, we now find all words in this dummy text:

text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
word_regex = "[A-Za-z]+" # All strings that are made out of a sequence of alphabetic characters

words = re.findall(word_regex, text) # In English: Find all words in the text
print (words)

# Another way to do the same thing is to split the text by space.

space_regex = "\s+" # A sequence of arbitrary lenght of whitespace cahracters (empty characters and newline)

words = re.split(space_regex,text)

print (words)

# The only difference is that now we also get the punctuation marks,
# because we are not looking for words, we are looking for everything
# that exists between whitespaces

# Now a slightly more complicated example. Let's find all strings of the form
# '<word><punctuation><number>' in the text and then get the number out of them.

punc_regex = "[^\s\da-zA-Z]" # Everyting which is not a whitespace, a digit or an alphabetic character

important_stuff = word_regex+punc_regex + digits_regex

text2 = \
    """
    seed-172836
    text te&xt bla bla 97g149ghuanf?89276*35872  000iunia723-178236
    29874fsoei*9832
    jois8329oim|2937ndfnoi i|au&182763
    """

matches = re.findall(important_stuff, text2)
print(matches) # All the matches of the form '<word><punctuation><number>'
for match in matches:
    word, number = re.split(punc_regex,match) # split match at punc
    print(number)






