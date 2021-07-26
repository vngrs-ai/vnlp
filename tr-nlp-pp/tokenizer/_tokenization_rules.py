#!/usr/local/bin/python
# -*- coding: utf-8 -*-

rules = [
'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$',  # email
'(?<=[^0-9])(\:)(?=[^0-9.]|[\s]|$)',           # time
'(\#)(?=[^a-z0-9]|[\s]|$)',                    # hashtag
'(?:(https?://)?(http?://)(www\.)?[a-z0-9]+\.[a-z][.a-z0-9\?=\-\+%&\_/])',      #handling web addresses
'\?+', '\!+', '\,', '\;', '\*', '\^',
r'\(|\)|\[|\]|\{|\}',
]

split = r'\s|\t|\n|\r'