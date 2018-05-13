# -*- coding: utf-8 -*-
import time
import arrow
import requests

from fake_useragent import UserAgent

# Request
# http://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date=20180501
#
# Response
# {
#   stat: "OK",
#   data[["107/05/02", "open", "high", "low", "close"],...]
# }
with open('weighted_stock_price_index.csv', 'w') as file:
    ua = UserAgent()

    # header
    file.write('"date","open","high","low","close"\n')

    start_date = arrow.get(2013, 1, 1)
    end_date = arrow.now()

    for d in arrow.Arrow.range('month', start_date, end_date):
        date = d.format('YYYYMMDD')
        print('quering {}'.format(date))

        url = 'http://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={}&_=1525951655386'.format(date)
        headers = {'User-Agent': ua.random}
        r = requests.get(url, headers=headers)

        if r.status_code != requests.codes.ok:
            print('error response {}'.format(r))
            exit()

        # convert to JSON object
        r = r.json()
        stat = r.get('stat')
        if stat == 'OK':
            data = r.get('data')
            for row in data:
                date, open, high, low, close = row
                #將民國年轉為西元年
                date = date.split('/')
                date[0] = str(int(date[0]) + 1911)
                date = ''.join(date)
                file.write('"{}","{}","{}","{}","{}"\n'.format(date, open, high, low, close))
        else:
            print('stat({}) is not OK'.format(stat))

        time.sleep(60)
