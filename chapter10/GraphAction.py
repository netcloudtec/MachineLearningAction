#Yahoo！PlaceFinder API
#导入urllib
import urllib
#导入json模块
import json

#利用地名，城市获取位置经纬度函数
def geoGrab(stAddress,city):
    #获取经纬度网址
    apiStem='http://where.yahooapis.com/geocode?'
    #初始化一个字典，存储相关参数
    params={}
    #返回类型为json
    params['flags']='J'
    #参数appid
    params['appid']='ppp68N8t'
    #参数地址位置信息
    # params['location']=('%s %s', %(stAddress,city))
    #利用urlencode函数将字典转为URL可以传递的字符串格式
    url_params=urllib.urlencode(params)
    #组成完整的URL地址api
    yahooApi=apiStem+url_params
    #打印该URL地址
    print('%s',yahooApi)
    #打开URL，返回json格式的数据
    c=urllib.urlopen(yahooApi)
    #返回json解析后的数据字典
    return json.load(c.read())

from time import sleep
#具体文本数据批量地址经纬度获取函数
def massPlaceFind(fileName):
    #新建一个可写的文本文件，存储地址，城市，经纬度等信息
    fw=open('./data/places.txt','wb+')
    #遍历文本的每一行
    for line in open(fileName).readlines():
        #去除首尾空格
        line =line.strip()
        #按tab键分隔开
        lineArr=line.split('\t')
        #利用获取经纬度函数获取该地址经纬度
        retDict=geoGrab(lineArr[1],lineArr[2])
        #如果错误编码为0，表示没有错误，获取到相应经纬度
        if retDict['ResultSet']['Error']==0:
            #从字典中获取经度
            lat=float(retDict['ResultSet']['Results'][0]['latitute'])
            #维度
            lng=float(retDict['ResultSet']['Results'][0]['longitute'])
            #打印地名及对应的经纬度信息
            # print('%s %f %f',%(lineArr[0],lat,lng))
            #将上面的信息存入新的文件中
            # fw.write('%s\t%f\t%f\n',%(line,lat,lng))
        #如果错误编码不为0，打印提示信息
        else:print('error fetching')
        #为防止频繁调用API，造成请求被封，使函数调用延迟一秒
        sleep(1)
    #文本写入关闭
    fw.close()


if __name__ == '__main__':
    massPlaceFind("./data/portlandClubs.txt")
