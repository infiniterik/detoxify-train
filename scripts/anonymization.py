import hashlib
import base64
import pandas as pd
import re

def remove_url(text):
    return re.sub(r'(https?://)?[-a-zA-Z0-9@:%._\+~#=]{1,256}(\.[a-z]{2,6})+\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', text)    

def hashCompUrl(text):
    modText = text
    prefix = ''
    if 'https://' in text:
        prefix = 'https://'
        text = text[8:]
    elif 'http://' in text:
        prefix = 'http://'
        text = text[7:]
    words = text.split('/')
    if len(words)==1:       # only one element 'www.leidos.com'
        return modText
    elif len(words)==2:     # two elements 'www.leidos.com/hash(elem2)'
        if len(words[0])>0:
            #print('Len=2 ModText' + modText)
            if len(words[1])>0: #www.leidos.com/data
                # leave elem1/words[0] as plaintext
                word1_h = md5Hash(words[1])
                modText = prefix + words[0] + '/' + word1_h
            else:   #www.leidos.com/
                modText = prefix + words[0]  + '/'
            #print('Len=2 ModText' + modText)
        else:   # empty element (http://)/ or (http://)/elem2 - corrupted entry
            if len(words[1])>0: #http:///data
                words1_h = md5Hash(words[1])
                modText = prefix + words[0] + '/' + words1_h
            else:   #http:///
                modText = prefix + md5Hash(text)
    elif len(words)==3:    # three elements 'www.leidos.com/hash(elem2)/hash(elem3)'
        if len(words[0])>0:
            #print('Len=3 ModText: ' + modText)
            if len(words[1])>0:  # elements not empty
                # leave elem1/words[0] as plaintext
                elem2 = md5Hash(words[1])
                if len(words[2])>0:     #www.leidos.com/data/data2
                    elem3 = md5Hash(words[2])
                    modText = prefix + words[0] + '/' + elem2 + '/' + elem3
                else:                   #www.leidos.com/data/
                    modText = prefix + words[0] + '/' + elem2 + '/' 
            else:  # www.leidos.com//something
                elem2 = ''
                if len(words[2])>0:     #www.leidos.com//data2
                    elem3 = md5Hash(words[2])
                    modText = prefix + words[0] + '/' + elem2 + '/' + elem3
                else:                   #www.leidos.com//
                    modText = prefix + words[0] + '/' + elem2 + '/'
            #print('Len=3 ModText: ' + modText)
        else:   # first element is corrupted entry http:///something/
            modText = prefix + md5Hash(text)
    elif len(words)>3:     # three elements + 'www.leidos.com/hash(elem2)/hash(elem3)/hash(remaining elems)
        if len(words[0])>0:
            #print('Len>3 ModText: ' + modText)
            if len(words[1])>0: # elements not empty
                # leave elem1/words[0] as plaintext
                elem2 = md5Hash(words[1])
            else:
                elem2 = ''
            if len(words[2])>0:
                elem3 = md5Hash(words[2])
            else:
                elem3 = ''
            lastElements = ''
            for i in range(3, len(words)):
                lastElements += words[i] + '/'
            lastElements = lastElements[:-1]
            hashLastElements = md5Hash(lastElements)
            modText = prefix + words[0] + '/' + elem2 + '/' + elem3 + '/' + hashLastElements
            #print('Len>3 ModText ' + modText)
        else:  # corrupted entry
            modText = prefix + md5Hash(text)
            #print modText
    else:# no username
        modText = md5Hash(modText)
    return modText    

def md5HashRd(text, elemLen):
    #This method hashes Reddit URLs by their components
    modText = text
    text = text[elemLen:]
    words = text.split('/')
    if len(words)==6: # r/subReddit/type/id/titleText/ (last empty)
        id_h = md5Hash(words[3])
        title_h = md5Hash(words[4])
        hashText = words[0] + '/' + words[1] + '/' + words[2] + '/' + id_h + '/' + title_h + '/'
        #print('hashText: ' + hashText)
        modText = modText.replace(text, hashText,1)
    elif len(words)==2: # r/jobs4bitcoins (user/channel name?)
        name_h = md5Hash(words[1])
        hashText = words[0] + '/' + name_h
        #plainText = words[0] + '/' + words[1]
        modText = modText.replace(text, hashText,1)
    else:   # not six or two components
        #print('Reddit Not six or two comp: ' + modText)
        modText = hashCompUrl(modText)
    return modText


def hashURL(mod):
    #Hash an input URL field
    if mod is None or mod == "" or mod == 'None':
        return ""
    elif mod == 'http:/' or mod == 'https:/':
        return ""
    trUrl = mod.strip()
        
    if 'reddit.com/' in trUrl:
        idx = trUrl.index('reddit.com/')
        hashUrl = md5HashRd(trUrl, idx + 11)
    elif 'https://redd.it/' in trUrl:
        hashUrl = md5HashRd(trUrl, 16)
    else:
        #hashUrl = hashCompUrl(trUrl)
        hashUrl = trUrl     ###Don't change other Urls 2/14/2020
        
    mod = mod.replace(trUrl, hashUrl)
    return mod


def md5Hash(text):
    #This method does the actual hashing of the input
    if text is None or text == "" or text == 'None':
        return ""
    else:
        #Use hex digest for 32 character md5 hash
        #return hashlib.md5(text.encode('utf-8')).hexdigest()
        #Use base64 encoding for 22 character md5 hash
        textHash = base64.urlsafe_b64encode(hashlib.md5(str(text).encode('utf-8')).digest()).decode('utf-8')
        textHash = textHash[:-2] #Last two characters are always '==' and not needed
        return textHash

def anonTwitchChat(row_d):
    row_d = row_d.to_dict()
    new_row = row_d.copy()
    h_fields = ['_id', 'channel_id']
    for field in h_fields:
        if field in row_d:
            if pd.notnull(row_d[field]):
                h_field_name = field + '_h'
                if 'url' in field:
                    anon_t = hashURL(row_d[field])
                else:
                    anon_t = md5Hash(row_d[field])
                new_row[h_field_name] = anon_t
            new_row.pop(field)
        
    if 'commenter' in new_row:
        commenter = new_row['commenter']
        h_fields = ['_id', 'display_name', 'name', 'logo']
        for field in h_fields:
            h_field_name = field + '_h'
            try:
                anon_t = md5Hash(commenter[field])
                commenter[h_field_name] = anon_t
                commenter.pop(field)
            except AttributeError as ae:
                print('Attribute Error, commenter: ' + str(commenter))
    return row_d, new_row



def anonYT(row_d):
    row_d = row_d.to_dict()
    new_row = row_d.copy()
    h_fields = ['author', 'snippet.authorDisplayName', 'snippet.authorProfileImageUrl', 'snippet.authorChannelUrl', 'snippet.authorChannelId.value', 
    'snippet.topLevelComment.snippet.authorDisplayName', 'snippet.topLevelComment.snippet.authorProfileImageUrl', 'snippet.topLevelComment.snippet.authorChannelUrl', 
    'snippet.topLevelComment.snippet.authorChannelId.value']
    for field in h_fields:
        if field in row_d:
            if pd.notnull(row_d[field]):
                h_field_name = field + '_h'
                if 'url' in field:
                    anon_t = hashURL(row_d[field])
                else:
                    anon_t = md5Hash(row_d[field])
                new_row[h_field_name] = anon_t
            new_row.pop(field)
    return row_d, new_row


def anonThread(row_d):
    row_d = row_d.to_dict()
    new_row = row_d.copy()
    h_fields = ['author', 'author_fullname', 'url', 'permalink', 'full_link', 'thread_author', 'thread_url', 'post_url', 'post_author', 'post_id', 
    'post_content', 'username','thread_id', 'thread_name', 'user_id', 'user_page']
    for field in h_fields:
        if field in row_d:
            if pd.notnull(row_d[field]):
                h_field_name = field + '_h'
                if 'url' in field:
                    anon_t = hashURL(row_d[field])
                else:
                    anon_t = md5Hash(row_d[field])
                new_row[h_field_name] = anon_t
            new_row.pop(field)
        
    if 'reactions' in new_row and len(new_row['reactions'])>0:
        reactions = new_row['reactions']
        for reaction in reactions:
            h_fields = ['username', 'user_id', 'user_page', 'post_id']
            for field in h_fields:
                h_field_name = field + '_h'
                try:
                    anon_t = md5Hash(reaction[field])
                    reactions[h_field_name] = anon_t
                    reaction.pop(field)
                except AttributeError as ae:
                    print('Attribute Error, reaction: ' + str(reaction))
    if 'linked_urls' in new_row:   ##
        linkedUrls = new_row['linked_urls']
        linkedUrls_h = []
        for linkedUrl in linkedUrls:
            linkedUrl_h = hashURL(linkedUrl)
            linkedUrls_h.append(linkedUrl_h)
        new_row['linked_urls_h'] = linkedUrls_h
        new_row.pop('linked_urls')   
    return row_d, new_row