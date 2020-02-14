import json

data = {'package_num':40, 'timeInterval':0.05}
json_str = json.dump(data, open('defined.json','w'))