import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
%matplotlib qt4
from dateutil.parser import parse
from IPython.display import HTML

dataset = pd.read_csv(u'F:\\Tianchi_power\\Tianchi_power.csv')
for key , group in dataset.groupby('user_id'):
    if group.shape[0]==609:
        print key,group.shape[0]
        break

group.drop('user_id', axis=1, inplace=True)


def getDayOfWeek(x):
    date_parse = parse(x)
    return date_parse.weekday()+1
    
dataset['day_of_week'] = dataset.record_date.apply(getDayOfWeek)
# dataset.set_index(dataset.record_date, drop=True)
dataset.to_csv(u'F:\\Tianchi_power\\Tianchi_power_Hanle_addDayOfWeek.csv', header=True, index=False)
everyday_sumPower = dataset.groupby(['record_date'],sort=False)[['power_consumption']].agg('sum')
# everyday_sumPower
everyday_sumPower.to_csv(u'F:\\Tianchi_power\\Tianchi_power_Hanle_sumByDate.csv', header=True,index_label=False)、

# %matplotlib inline
chart_head_html = u"""
<div id="chart" style="width:800px; height:600px;"></div>
<script>
    require.config({
         paths:{
            echarts: '//cdn.bootcss.com/echarts/3.2.3/echarts.min',
         }
    });
    require(['echarts'],function(ec){
    var myChart = ec.init(document.getElementById('chart'));
    var option = {
        title: {
            text: '历史总用电量',
            subtext: '扬州'
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: '总用电量'
        },
        toolbox: {
            show: true,
            feature: {
                dataZoom: {
                    yAxisIndex: 'none'
                },
                dataView: {readOnly: false},
                magicType: {type: ['line', 'bar']},
                restore: {},
                saveAsImage: {}
            }
        },
        xAxis:  {
            type: 'category',
            boundaryGap: false,
            data: %s""" % str(list(everyday_sumPower.index))

chart_mo_html = u"""},
        yAxis: {
            type: 'value',
            axisLabel: {
                formatter: '{value}'
            }
        },
        series: [
            {
                name:'当日总用电量',
                type:'line',
                data: %s""" % str(list(everyday_sumPower.power_consumption))
chart_edn_html = u""",
                markPoint: {
                    data: [
                        {type: 'max', name: '最大值'},
                        {type: 'min', name: '最小值'}
                    ]
                },
                markLine: {
                    data: [
                        {type: 'average', name: '平均值'}
                    ]
                }
            }
        ]
    };
    myChart.setOption(option);
    });
</script>"""
html = HTML(chart_head_html+chart_mo_html+chart_edn_html)