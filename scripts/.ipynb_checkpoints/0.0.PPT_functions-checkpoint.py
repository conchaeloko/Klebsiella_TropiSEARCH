# PPT_functions.py


def get_full_entry(IPR , pprint_o_nah) :
    import pprint
    pp = pprint.PrettyPrinter(width = 150)
    for index_i, entry in enumerate(xml_interpro['interprodb']["interpro"]) :
        if entry["@id"] == IPR :
            output = xml_interpro['interprodb']["interpro"][index_i]
    if pprint_o_nah == True :
        return pp.pprint(output)
    else :
        return output
    
def all_strings(list_p) :
    for index_list, element in enumerate(list_p) :
        if isinstance(element , str) == False :
            return False
            break
        else :
            continue
    else :
        return True
    
    
def any_string(list_p) :
    for index_list, element in enumerate(list_p) :
        if isinstance(element , str) == True :
            return True
            break
    else :
        return False
    
def return_string(list_p) :
    string = str()
    for index_list, element in enumerate(list_p) :
        if isinstance(element , str) == True :
            string = string + element
        else :
            continue
def good_term(string) :
    for term in depo_terms :
        if string.lower().count(term) >0 :
            return True
            break
    else :
        return False