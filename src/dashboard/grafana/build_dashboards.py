"""
Grafana dashboards as code
==========================
Generates the provisioned dashboards in ./dashboards from one definition, so
queries, colours, thresholds and filters stay consistent across dashboards.

    python -m src.dashboard.grafana.build_dashboards          # regenerate JSON
    python -m src.dashboard.grafana.build_dashboards --check  # fail if JSON is stale

Dashboards (linked to each other, all in the "Crop Disease" folder):
    crop-overview   Field Overview      live risk, map, per-station status, class timeline
    crop-spectral   Spectral Analysis   band reflectance and vegetation indices over time
    crop-models     Model Performance   live accuracy/calibration + offline pipeline results
    crop-system     System Health       ingest rate, freshness, bridge counters

Data comes from InfluxDB measurements written by:
    src.dashboard.mqtt_subscriber           spectral_readings, disease_predictions, bridge_stats
    src.dashboard.seed_history              same measurements, source="backfill"
    src.dashboard.publish_pipeline_metrics  model_evaluation, training_history, cross_validation,
                                            feature_selection, model_significance, batch_risk,
                                            pipeline_run
"""

import argparse
import json
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(__file__), 'dashboards')
DS = {'type': 'influxdb', 'uid': 'influxdb-crop'}
TAG = 'crop'

CLASS_COLORS = {0: ('Healthy', 'green'), 1: ('Early Stress', 'orange'),
                2: ('Active Disease', 'red')}
RISK_STEPS = [(None, 'green'), (0.2, 'yellow'), (0.5, 'orange'), (0.7, 'red')]

# ── Flux helpers ──────────────────────────────────────────────

RANGE = '  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)'
LIVE_FILTER = ('  |> filter(fn: (r) => r.station_id =~ /^${station:regex}$/ '
               'and r.source =~ /^${source:regex}$/)')
RUN_FILTER = '  |> filter(fn: (r) => r.run_id == "${run_id}")'


def flux(measurement: str, fields=(), live: bool = True, rng: str = RANGE, *more: str) -> str:
    """from(bucket) |> range |> measurement [|> fields] [|> station/source filter] [|> more]."""
    lines = ['from(bucket: "${bucket}")', rng,
             f'  |> filter(fn: (r) => r._measurement == "{measurement}")']
    if fields:
        cond = ' or '.join(f'r._field == "{f}"' for f in fields)
        lines.append(f'  |> filter(fn: (r) => {cond})')
    if live:
        lines.append(LIVE_FILTER)
    lines.extend(f'  |> {m}' for m in more)
    return '\n'.join(lines)


def offline(measurement: str, fields=(), *more: str) -> str:
    """Pipeline results for the selected run (independent of the dashboard time range)."""
    return flux(measurement, fields, False, '  |> range(start: -3650d)',
                *(RUN_FILTER.removeprefix('  |> '),), *more)


def latest_per_station(fields, measurement='disease_predictions') -> str:
    return flux(measurement, fields, True, RANGE,
                'group(columns: ["station_id", "_field"])', 'last()',
                'pivot(rowKey: ["station_id", "_time"], columnKey: ["_field"], '
                'valueColumn: "_value")',
                'group()')


def epoch_curves(curves: dict) -> str:
    """
    Training curves for the selected run as a single frame:
    epoch | "<model> <label>" ... (the trend panel accepts only one frame).

    curves: {label: training_history field}, e.g. {'train': 'train_loss'}.
    """
    fields = ['epoch', *curves.values()]
    cond = ' or '.join(f'r._field == "{f}"' for f in fields)
    lines = ['data = from(bucket: "${bucket}")',
             '  |> range(start: -3650d)',
             '  |> filter(fn: (r) => r._measurement == "training_history")',
             f'  |> filter(fn: (r) => {cond})',
             RUN_FILTER,
             '  |> pivot(rowKey: ["_time", "model"], columnKey: ["_field"], '
             'valueColumn: "_value")',
             '  |> group()']
    names = []
    for i, (label, field) in enumerate(curves.items()):
        name = f's{i}'
        names.append(name)
        lines.append(f'{name} = data |> map(fn: (r) => ({{epoch: r.epoch, '
                     f'series: r.model + " {label}", value: r.{field}}}))')
    lines += [f'union(tables: [{", ".join(names)}])',
              '  |> pivot(rowKey: ["epoch"], columnKey: ["series"], valueColumn: "value")',
              '  |> sort(columns: ["epoch"])']
    return '\n'.join(lines)


def target(query: str, ref: str = 'A') -> dict:
    return {'datasource': DS, 'refId': ref, 'query': query}


# ── Field config helpers ──────────────────────────────────────

def thresholds(steps=RISK_STEPS) -> dict:
    return {'mode': 'absolute',
            'steps': [{'color': c, 'value': v} for v, c in steps]}


def class_name_colors() -> list:
    """Overrides that colour series/fields named after the classes."""
    return [{'matcher': {'id': 'byName', 'options': name},
             'properties': [{'id': 'color', 'value': {'mode': 'fixed', 'fixedColor': color}}]}
            for name, color in CLASS_COLORS.values()]


def rename(field: str, title: str, *props) -> dict:
    return {'matcher': {'id': 'byName', 'options': field},
            'properties': [{'id': 'displayName', 'value': title}, *props]}


def prop(pid: str, value) -> dict:
    return {'id': pid, 'value': value}


def risk_cell() -> list:
    return [prop('custom.cellOptions', {'type': 'color-background', 'mode': 'gradient'}),
            prop('thresholds', thresholds()), prop('color', {'mode': 'thresholds'}),
            prop('min', 0), prop('max', 1), prop('decimals', 2)]


# ── Panel builders ────────────────────────────────────────────

class Board:
    """Collects panels and lays them out on Grafana's 24-column grid."""

    def __init__(self, uid: str, title: str, description: str, variables: list,
                 time_from: str = 'now-6h', refresh: str = '30s'):
        self.uid, self.title, self.description = uid, title, description
        self.variables, self.time_from, self.refresh = variables, time_from, refresh
        self.panels, self._y, self._x, self._row_h, self._id = [], 0, 0, 0, 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def row(self, title: str) -> None:
        self._newline()
        self.panels.append({'type': 'row', 'title': title, 'collapsed': False, 'panels': [],
                            'id': self._next_id(),
                            'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': self._y}})
        self._y += 1

    def _newline(self) -> None:
        if self._x:
            self._y += self._row_h
            self._x, self._row_h = 0, 0

    def add(self, panel: dict, w: int, h: int) -> dict:
        if self._x + w > 24:
            self._newline()
        panel['id'] = self._next_id()
        panel['gridPos'] = {'h': h, 'w': w, 'x': self._x, 'y': self._y}
        panel.setdefault('datasource', DS)
        self.panels.append(panel)
        self._x += w
        self._row_h = max(self._row_h, h)
        return panel

    def to_json(self) -> dict:
        return {
            'uid': self.uid, 'title': self.title, 'description': self.description,
            'tags': [TAG], 'editable': True, 'graphTooltip': 1, 'schemaVersion': 39,
            'version': 1, 'timezone': 'browser', 'refresh': self.refresh,
            'time': {'from': self.time_from, 'to': 'now'},
            'fiscalYearStartMonth': 0, 'liveNow': False,
            'annotations': {'list': [{
                'builtIn': 1, 'datasource': {'type': 'grafana', 'uid': '-- Grafana --'},
                'enable': True, 'hide': True, 'iconColor': 'rgba(0, 211, 255, 1)',
                'name': 'Annotations & Alerts', 'type': 'dashboard'}]},
            'links': [{'type': 'dashboards', 'tags': [TAG], 'asDropdown': False,
                       'includeVars': True, 'keepTime': True, 'title': 'Crop dashboards'}],
            'templating': {'list': self.variables},
            'panels': self.panels,
        }


def base(ptype: str, title: str, targets: list, description: str = '', **extra) -> dict:
    panel = {'type': ptype, 'title': title, 'description': description, 'targets': targets,
             'fieldConfig': {'defaults': {}, 'overrides': []}, 'options': {}}
    panel.update(extra)
    return panel


def stat(title, query, unit='none', steps=None, decimals=None, description='',
         no_value='0', color_mode='background', graph='none', calc='lastNotNull',
         fields=''):
    defaults = {'unit': unit, 'noValue': no_value, 'color': {'mode': 'thresholds'},
                'thresholds': thresholds(steps or [(None, 'blue')])}
    if decimals is not None:
        defaults['decimals'] = decimals
    return base('stat', title, [target(query)], description,
                fieldConfig={'defaults': defaults, 'overrides': []},
                options={'reduceOptions': {'calcs': [calc], 'fields': fields,
                                           'values': False},
                         'colorMode': color_mode, 'graphMode': graph, 'textMode': 'auto',
                         'justifyMode': 'center', 'orientation': 'auto', 'wideLayout': True,
                         'showPercentChange': False})


def timeseries(title, queries, unit='none', description='', display_name=None, steps=None,
               threshold_style='off', fill=10, stack=False, draw='line', min_=None, max_=None,
               overrides=None, points='auto', legend_calcs=('mean', 'lastNotNull'),
               interpolation=None):
    custom = {'drawStyle': draw, 'lineWidth': 2, 'fillOpacity': fill, 'gradientMode': 'opacity',
              'showPoints': points, 'pointSize': 5, 'spanNulls': True,
              'lineInterpolation': interpolation or ('smooth' if draw == 'line' else 'linear'),
              'stacking': {'mode': 'normal' if stack else 'none', 'group': 'A'},
              'thresholdsStyle': {'mode': threshold_style}, 'axisSoftMin': min_,
              'axisSoftMax': max_, 'barAlignment': 0}
    defaults = {'unit': unit, 'custom': custom, 'color': {'mode': 'palette-classic'},
                'thresholds': thresholds(steps or [(None, 'green')])}
    if display_name:
        defaults['displayName'] = display_name
    if min_ is not None:
        defaults['min'] = min_
    if max_ is not None:
        defaults['max'] = max_
    targets = [target(q, chr(65 + i)) for i, q in enumerate(
        queries if isinstance(queries, (list, tuple)) else [queries])]
    return base('timeseries', title, targets, description,
                fieldConfig={'defaults': defaults, 'overrides': overrides or []},
                options={'legend': {'displayMode': 'table', 'placement': 'bottom',
                                    'calcs': list(legend_calcs), 'showLegend': True},
                         'tooltip': {'mode': 'multi', 'sort': 'desc'}})


def table(title, query, overrides=None, description='', sort_by=None, unit='none'):
    return base('table', title, [target(query)], description,
                fieldConfig={'defaults': {'unit': unit, 'custom': {
                    'align': 'auto', 'cellOptions': {'type': 'auto'}, 'filterable': False},
                    'color': {'mode': 'thresholds'}, 'thresholds': thresholds([(None, 'text')])},
                    'overrides': overrides or []},
                options={'showHeader': True, 'cellHeight': 'sm', 'footer': {'show': False},
                         'sortBy': sort_by or []})


def geomap(title, query, color_field, text_field=None, description='', size=9,
           steps=RISK_STEPS, zoom=10):
    style = {'size': {'fixed': size, 'min': 4, 'max': 20},
             'color': {'field': color_field, 'fixed': 'dark-green'},
             'opacity': 0.8,
             'symbol': {'mode': 'fixed', 'fixed': 'img/icons/marker/circle.svg'},
             'rotation': {'fixed': 0, 'mode': 'mod', 'min': -360, 'max': 360},
             'symbolAlign': {'horizontal': 'center', 'vertical': 'center'},
             'textConfig': {'fontSize': 12, 'offsetX': 0, 'offsetY': -16,
                            'textAlign': 'center', 'textBaseline': 'middle'}}
    if text_field:
        style['text'] = {'mode': 'field', 'field': text_field, 'fixed': ''}
    return base('geomap', title, [target(query)], description,
                fieldConfig={'defaults': {'color': {'mode': 'thresholds'},
                                          'thresholds': thresholds(steps), 'min': 0, 'max': 1,
                                          'custom': {'hideFrom': {'legend': False,
                                                                  'tooltip': False,
                                                                  'viz': False}}},
                             'overrides': []},
                options={'view': {'id': 'fit', 'lat': 12.97, 'lon': 77.59, 'zoom': zoom,
                                  'allLayers': True},
                         'controls': {'showZoom': True, 'mouseWheelZoom': True,
                                      'showAttribution': True, 'showScale': True},
                         'basemap': {'type': 'osm-standard', 'name': 'OpenStreetMap',
                                     'config': {}},
                         'layers': [{'type': 'markers', 'name': title, 'tooltip': True,
                                     'config': {'showLegend': True, 'style': style},
                                     'location': {'mode': 'coords', 'latitude': 'latitude',
                                                  'longitude': 'longitude'}}],
                         'tooltip': {'mode': 'details'}})


def text_state_mappings(states: dict) -> list:
    """Value mappings for text states: {text: colour}."""
    return [{'type': 'value', 'options': {
        text: {'color': color, 'index': i} for i, (text, color) in enumerate(states.items())}}]


CLASS_STATES = {name: color for name, color in CLASS_COLORS.values()}


def state_timeline(title, query, states=None, description=''):
    """State timeline over TEXT values, coloured by `states` ({text: colour})."""
    return base('state-timeline', title, [target(query)], description,
                fieldConfig={'defaults': {'mappings': text_state_mappings(states or CLASS_STATES),
                                          'color': {'mode': 'fixed', 'fixedColor': 'text'},
                                          'displayName': '${__field.labels.station_id}',
                                          'custom': {'lineWidth': 0, 'fillOpacity': 85}},
                             'overrides': []},
                options={'mergeValues': True, 'showValue': 'auto', 'alignValue': 'center',
                         'rowHeight': 0.85, 'legend': {'showLegend': True,
                                                       'displayMode': 'list',
                                                       'placement': 'bottom'},
                         'tooltip': {'mode': 'single'}})


def piechart(title, query, description='', overrides=None, display_name=None):
    defaults = {'color': {'mode': 'palette-classic'}}
    if display_name:
        defaults['displayName'] = display_name
    return base('piechart', title, [target(query)], description,
                fieldConfig={'defaults': defaults,
                             'overrides': overrides or []},
                options={'pieType': 'donut', 'displayLabels': ['percent'],
                         'reduceOptions': {'calcs': ['lastNotNull'], 'fields': '',
                                           'values': False},
                         'legend': {'displayMode': 'table', 'placement': 'right',
                                    'values': ['value', 'percent'], 'showLegend': True},
                         'tooltip': {'mode': 'single'}})


def bargauge(title, query, unit='none', steps=RISK_STEPS, min_=0, max_=1, description='',
             display_name=None):
    defaults = {'unit': unit, 'min': min_, 'max': max_, 'color': {'mode': 'thresholds'},
                'thresholds': thresholds(steps), 'decimals': 2}
    if display_name:
        defaults['displayName'] = display_name
    return base('bargauge', title, [target(query)], description,
                fieldConfig={'defaults': defaults, 'overrides': []},
                options={'displayMode': 'gradient', 'orientation': 'horizontal',
                         'showUnfilled': True, 'valueMode': 'color', 'namePlacement': 'top',
                         'sizing': 'auto', 'minVizHeight': 16, 'maxVizHeight': 300,
                         'reduceOptions': {'calcs': ['lastNotNull'], 'fields': '',
                                           'values': False}})


def barchart(title, query, x_field, unit='none', description='', overrides=None,
             min_=None, max_=None, show_value='always'):
    defaults = {'unit': unit, 'color': {'mode': 'palette-classic'}, 'decimals': 3,
                'custom': {'fillOpacity': 80, 'lineWidth': 1, 'gradientMode': 'hue',
                           'axisSoftMin': min_, 'axisSoftMax': max_}}
    return base('barchart', title, [target(query)], description,
                fieldConfig={'defaults': defaults, 'overrides': overrides or []},
                options={'xField': x_field, 'orientation': 'auto', 'groupWidth': 0.75,
                         'barWidth': 0.9, 'showValue': show_value, 'stacking': 'none',
                         'xTickLabelRotation': 0, 'xTickLabelSpacing': 0,
                         'legend': {'displayMode': 'list', 'placement': 'bottom',
                                    'showLegend': True},
                         'tooltip': {'mode': 'multi'}})


def trend(title, query, x_field, unit='none', description='', min_=None, max_=None):
    custom = {'drawStyle': 'line', 'lineWidth': 2, 'fillOpacity': 0, 'showPoints': 'auto',
              'pointSize': 4, 'spanNulls': True, 'lineInterpolation': 'smooth',
              'axisSoftMin': min_, 'axisSoftMax': max_}
    return base('trend', title, [target(query)], description,
                fieldConfig={'defaults': {'unit': unit, 'custom': custom,
                                          'color': {'mode': 'palette-classic'}},
                             'overrides': []},
                options={'xField': x_field,
                         'legend': {'displayMode': 'list', 'placement': 'bottom',
                                    'showLegend': True},
                         'tooltip': {'mode': 'multi'}})


def histogram(title, query, unit='none', description='', bucket_size=None, display_name=None,
              overrides=None, fill=70):
    defaults = {'unit': unit, 'color': {'mode': 'palette-classic'},
                'custom': {'fillOpacity': fill, 'lineWidth': 1, 'gradientMode': 'none'}}
    if display_name:
        defaults['displayName'] = display_name
    options = {'combine': False, 'legend': {'displayMode': 'list', 'placement': 'bottom',
                                            'showLegend': True}}
    if bucket_size:
        options['bucketSize'] = bucket_size
    return base('histogram', title, [target(query)], description,
                fieldConfig={'defaults': defaults, 'overrides': overrides or []},
                options=options)


def heatmap(title, query, description='', unit='none'):
    return base('heatmap', title, [target(query)], description,
                fieldConfig={'defaults': {'custom': {'hideFrom': {'legend': False,
                                                                  'tooltip': False,
                                                                  'viz': False},
                                                     'scaleDistribution': {'type': 'linear'}}},
                             'overrides': []},
                options={'calculate': True,
                         'calculation': {'yBuckets': {'mode': 'count', 'value': '16'}},
                         'color': {'mode': 'scheme', 'scheme': 'Greens', 'exponent': 0.5,
                                   'steps': 64, 'fill': 'dark-green', 'reverse': False},
                         'cellGap': 1, 'yAxis': {'axisPlacement': 'left', 'unit': unit},
                         'rowsFrame': {'layout': 'auto'}, 'showValue': 'never',
                         'tooltip': {'mode': 'single', 'yHistogram': True, 'showColorScale': False},
                         'legend': {'show': True}, 'exemplars': {'color': 'rgba(255,0,255,0.7)'},
                         'filterValues': {'le': 1e-9}})


def text(title, markdown):
    return {'type': 'text', 'title': title, 'options': {'mode': 'markdown', 'content': markdown},
            'datasource': None, 'targets': []}


def alertlist(title):
    return {'type': 'alertlist', 'title': title, 'datasource': None, 'targets': [],
            'options': {'viewMode': 'list', 'groupMode': 'default', 'groupBy': [],
                        'maxItems': 10, 'sortOrder': 1, 'dashboardAlerts': False,
                        'alertName': '', 'alertInstanceLabelFilter': '', 'folder': None,
                        'showInstances': True,
                        'stateFilter': {'firing': True, 'pending': True, 'noData': True,
                                        'normal': False, 'error': True}}}


# ── Variables ─────────────────────────────────────────────────

def constant(name, value):
    return {'name': name, 'type': 'constant', 'query': value, 'hide': 2, 'skipUrlSync': True,
            'current': {'text': value, 'value': value}, 'options': [{'text': value,
                                                                     'value': value,
                                                                     'selected': True}]}


def tag_variable(name, label, tag, measurement, include_all=True, sort=1, start='-30d'):
    query = ('import "influxdata/influxdb/schema"\n'
             f'schema.tagValues(bucket: "${{bucket}}", tag: "{tag}", '
             f'predicate: (r) => r._measurement == "{measurement}", start: {start})')
    var = {'name': name, 'label': label, 'type': 'query', 'datasource': DS, 'query': query,
           'definition': query, 'refresh': 2, 'sort': sort, 'hide': 0, 'options': [],
           'multi': include_all, 'includeAll': include_all}
    if include_all:
        var['allValue'] = '.*'
        var['current'] = {'selected': True, 'text': ['All'], 'value': ['$__all']}
    else:
        var['current'] = {}
    return var


LIVE_VARS = [constant('bucket', 'crop_data'),
             tag_variable('station', 'Station', 'station_id', 'spectral_readings'),
             tag_variable('source', 'Source', 'source', 'spectral_readings')]


# ── Dashboards ────────────────────────────────────────────────

def overview() -> Board:
    b = Board('crop-overview', 'Crop Disease — Field Overview',
              'Live disease risk across sensor stations: KPIs, map, latest status per '
              'station and how predictions evolve over time.', LIVE_VARS)

    b.row('Field status')
    b.add(stat('Stations reporting', flux(
        'spectral_readings', ['nir_842'], True, RANGE, 'group(columns: ["station_id"])',
        'last()', 'group()', 'count()'), steps=[(None, 'blue')],
        description='Stations with at least one reading in the selected time range'), 4, 4)
    b.add(stat('Readings', flux('spectral_readings', ['nir_842'], True, RANGE, 'group()',
                                'count()'), unit='short', steps=[(None, 'blue')]), 4, 4)
    b.add(stat('Predictions', flux('disease_predictions', ['risk_score'], True, RANGE,
                                   'group()', 'count()'), unit='short',
               steps=[(None, 'blue')],
               description='One prediction per completed 30-reading crop cycle'), 4, 4)
    b.add(stat('Mean risk', flux('disease_predictions', ['risk_score'], True, RANGE, 'group()',
                                 'mean()'), unit='percentunit', decimals=1, steps=RISK_STEPS,
               no_value='-', description='Probability-weighted risk, averaged over '
                                         'all predictions in range'), 4, 4)
    b.add(stat('High-risk stations', flux(
        'disease_predictions', ['risk_score'], True, RANGE, 'group(columns: ["station_id"])',
        'last()', 'filter(fn: (r) => r._value > 0.7)', 'group()', 'count()'),
        steps=[(None, 'green'), (1, 'red')],
        description='Stations whose latest prediction has risk > 0.7'), 4, 4)
    b.add(stat('Live agreement with ground truth', flux(
        'disease_predictions', ['correct'], True, RANGE, 'group()', 'mean()'),
        unit='percentunit', decimals=1, no_value='n/a',
        steps=[(None, 'red'), (0.8, 'orange'), (0.9, 'green')],
        description='Share of live predictions matching the time-relative CDM label of '
                    'the replayed sequence (only for replayed stations)'), 4, 4)

    b.row('Where and what')
    b.add(geomap('Station risk map (latest prediction)', latest_per_station(
        ['risk_score', 'latitude', 'longitude', 'confidence', 'label_name']),
        'risk_score', 'station_id', zoom=11,
        description='Marker colour = latest probability-weighted risk'), 12, 11)
    latest = b.add(table('Latest prediction per station', latest_per_station(
        ['label_name', 'confidence', 'risk_score']) +
        '\n  |> keep(columns: ["station_id", "label_name", "risk_score", "confidence", '
        '"_time"])',
        overrides=[rename('station_id', 'Station', prop('custom.width', 90)),
                   rename('label_name', 'Prediction', prop('custom.cellOptions', {
                       'type': 'color-text'}), prop('mappings', text_state_mappings(
                           CLASS_STATES)), prop('custom.width', 150)),
                   rename('risk_score', 'Risk', *risk_cell(), prop('unit', 'percentunit'),
                          prop('decimals', 0), prop('custom.width', 110)),
                   rename('confidence', 'Confidence', prop('unit', 'percentunit'),
                          prop('decimals', 1), prop('custom.width', 110)),
                   rename('_time', 'Predicted at')],
        sort_by=[{'displayName': 'Risk', 'desc': True}]), 12, 11)
    latest['transformations'] = [{'id': 'organize', 'options': {'indexByName': {
        'station_id': 0, 'label_name': 1, 'risk_score': 2, 'confidence': 3, '_time': 4}}}]

    b.row('Over time')
    b.add(timeseries('Disease risk by station', flux(
        'disease_predictions', ['risk_score'], True, RANGE,
        'keep(columns: ["_time", "_value", "station_id"])'),
        unit='percentunit', display_name='${__field.labels.station_id}', min_=0, max_=1,
        steps=RISK_STEPS, threshold_style='dashed', points='always', fill=0,
        interpolation='stepAfter',
        description='Each point is one completed crop cycle; the risk holds until the '
                    'station completes its next cycle'), 16, 9)
    b.add(bargauge('Mean risk by station', flux(
        'disease_predictions', ['risk_score'], True, RANGE, 'group(columns: ["station_id"])',
        'mean()', 'keep(columns: ["_value", "station_id"])'), unit='percentunit',
        display_name='${__field.labels.station_id}'), 8, 9)
    b.add(state_timeline('Predicted class timeline', flux(
        'disease_predictions', ['label_name'], True, RANGE,
        'keep(columns: ["_time", "_value", "station_id"])'),
        description='Predicted crop state per station; each colour block lasts until the '
                    'next completed cycle'), 16, 8)
    b.add(piechart('Prediction mix', flux(
        'disease_predictions', ['label_name'], True, RANGE,
        'map(fn: (r) => ({r with label: r._value}))', 'group(columns: ["label"])', 'count()',
        'map(fn: (r) => ({_value: r._value, label: r.label}))'),
        display_name='${__field.labels.label}', overrides=class_name_colors()), 8, 8)
    b.add(timeseries('Class probabilities (mean across selected stations)', flux(
        'disease_predictions', ['prob_class_0', 'prob_class_1', 'prob_class_2'], True, RANGE,
        'map(fn: (r) => ({r with _field: if r._field == "prob_class_0" then "Healthy" '
        'else if r._field == "prob_class_1" then "Early Stress" else "Active Disease"}))',
        'group(columns: ["_field"])',
        'aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)',
        'keep(columns: ["_time", "_value", "_field"])'),
        unit='percentunit', stack=True, fill=60, min_=0, max_=1,
        overrides=class_name_colors()), 16, 8)
    b.add(alertlist('Active alerts'), 8, 8)
    return b


def spectral() -> Board:
    b = Board('crop-spectral', 'Crop Disease — Spectral Analysis',
              'Raw band reflectance and derived vegetation indices over time. Disease shows '
              'up as falling red-edge/NIR reflectance and NDVI/NDRE, and rising MSI.',
              LIVE_VARS)
    window = 'aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)'
    by_station = 'keep(columns: ["_time", "_value", "station_id"])'

    b.row('Band reflectance')
    bands = b.add(timeseries('Band reflectance — $station', flux(
        'spectral_readings', ['red_edge_705', 'red_edge_740', 'red_edge_783', 'nir_842',
                              'swir_1610', 'swir_2190'], True, RANGE,
        window, 'keep(columns: ["_time", "_value", "_field"])'),
        unit='percentunit', min_=0, max_=1, fill=0,
        description='Reflectance of each band (0-1). One panel per selected station.'), 12, 8)
    bands['repeat'] = 'station'
    bands['repeatDirection'] = 'h'
    bands['maxPerRow'] = 2

    b.row('Vegetation indices by station')
    for field, title, unit, desc in (
        ('NDVI', 'NDVI (red-edge)', 'none', 'Falls as chlorophyll / canopy degrades'),
        ('NDRE', 'NDRE', 'none', 'Chlorophyll-sensitive; early stress indicator'),
        ('MSI', 'Moisture Stress Index', 'none', 'Rises with water stress (SWIR / NIR)'),
        ('REIP_approx', 'Red-edge inflection point', 'suffix: nm', 'Shifts towards shorter '
                                                               'wavelengths under stress'),
        ('SWIR_ratio', 'SWIR ratio (2190 / 1610)', 'none', 'Water / structure changes'),
        ('CRI', 'Chlorophyll red-edge index', 'none', '1/R705 - 1/R740'),
    ):
        b.add(timeseries(title, flux('spectral_readings', [field], True, RANGE,
                                     'timedMovingAverage(every: v.windowPeriod, period: 10m)',
                                     by_station),
                         unit=unit, display_name='${__field.labels.station_id}',
                         description=desc + ' (10-minute rolling mean)', fill=0,
                         legend_calcs=('lastNotNull',)), 8, 8)

    b.row('Distributions')
    b.add(heatmap('NDVI distribution over time (selected stations)', flux(
        'spectral_readings', ['NDVI'], True, RANGE, 'group()',
        'keep(columns: ["_time", "_value"])'),
        description='Density of NDVI readings per time bucket'), 12, 9)
    b.add(histogram('NDVI by true label', flux(
        'spectral_readings', ['NDVI'], True, RANGE, 'group(columns: ["profile"])',
        'keep(columns: ["_time", "_value", "profile"])'),
        display_name='${__field.labels.profile}', bucket_size=0.01, fill=35,
        overrides=[{'matcher': {'id': 'byName', 'options': profile},
                    'properties': [prop('displayName', name),
                                   prop('color', {'mode': 'fixed', 'fixedColor': color})]}
                   for profile, (name, color) in zip(
                       ('healthy', 'early_stress', 'active_disease'), CLASS_COLORS.values())],
        description='How well NDVI alone separates the time-relative CDM labels '
                    'of the replayed real sequences'), 12, 9)
    b.add(table('Latest indices per station', latest_per_station(
        ['NDVI', 'NDRE', 'MSI', 'REIP_approx'], 'spectral_readings') +
        '\n  |> keep(columns: ["station_id", "_time", "NDVI", "NDRE", "MSI", "REIP_approx"])',
        overrides=[rename('station_id', 'Station'), rename('_time', 'Last reading'),
                   rename('NDVI', 'NDVI', prop('custom.cellOptions', {
                       'type': 'gauge', 'mode': 'gradient', 'valueDisplayMode': 'text'}),
                       prop('min', 0), prop('max', 1), prop('decimals', 3),
                       prop('color', {'mode': 'continuous-RdYlGr'})),
                   rename('NDRE', 'NDRE', prop('custom.cellOptions', {
                       'type': 'gauge', 'mode': 'gradient', 'valueDisplayMode': 'text'}),
                       prop('min', 0), prop('max', 0.5), prop('decimals', 3),
                       prop('color', {'mode': 'continuous-RdYlGr'})),
                   rename('MSI', 'MSI', prop('decimals', 3)),
                   rename('REIP_approx', 'REIP (nm)', prop('decimals', 1))]), 24, 7)
    return b


def models() -> Board:
    run_var = tag_variable('run_id', 'Pipeline run', 'run_id', 'pipeline_run',
                           include_all=False, sort=2, start='-3650d')
    b = Board('crop-models', 'Crop Disease — Model Performance',
              'Live model behaviour on replayed real Sentinel-2 sequences (agreement with '
              'their CDM labels, confidence, out-of-distribution rate) and the offline '
              'evaluation published by run_pipeline.py (CDM + Sentinel-2, weak labels).',
              [*LIVE_VARS, run_var], time_from='now-24h', refresh='1m')

    b.row('Live model (selected time range)')
    b.add(timeseries('Rolling agreement with ground truth (last 20 predictions)', flux(
        'disease_predictions', ['correct'], True, RANGE, 'group()',
        'sort(columns: ["_time"])', 'movingAverage(n: 20)',
        'keep(columns: ["_time", "_value"])'),
        unit='percentunit', min_=0, max_=1, display_name='Agreement',
        steps=[(None, 'red'), (0.8, 'orange'), (0.9, 'green')], threshold_style='area',
        fill=0), 12, 8)
    b.add(table('Live confusion matrix (rows = true profile)', flux(
        'disease_predictions', ['label_name'], True, RANGE,
        'filter(fn: (r) => exists r.true_profile)',
        'map(fn: (r) => ({r with predicted: r._value}))',
        'group(columns: ["true_profile", "predicted"])', 'count()', 'group()',
        'pivot(rowKey: ["true_profile"], columnKey: ["predicted"], valueColumn: "_value")'),
        overrides=[rename('true_profile', 'True profile')],
        description='Counts of live predictions by CDM label of the replayed sequence '
                    '(row) and model output (column)'), 12, 8)
    b.add(histogram('Prediction confidence', flux(
        'disease_predictions', ['confidence'], True, RANGE, 'group()',
        'keep(columns: ["_time", "_value"])'), unit='percentunit', bucket_size=0.02,
        display_name='Confidence'), 8, 8)
    b.add(timeseries('Mean confidence and risk', [
        flux('disease_predictions', ['confidence', 'risk_score'], True, RANGE,
             'group(columns: ["_field"])',
             'aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)',
             'keep(columns: ["_time", "_value", "_field"])')],
        unit='percentunit', min_=0, max_=1, fill=0, points='always'), 8, 8)
    b.add(stat('Out-of-distribution predictions', flux(
        'disease_predictions', ['out_of_distribution'], True, RANGE,
        'filter(fn: (r) => r._value == true)', 'group()', 'count()'),
        steps=[(None, 'green'), (1, 'red')],
        description='Inputs with any standardized feature beyond |z| > 8'), 4, 8)
    b.add(stat('Live predictions with ground truth', flux(
        'disease_predictions', ['correct'], True, RANGE, 'group()', 'count()'),
        unit='short'), 4, 8)

    b.row('Offline evaluation — pipeline run $run_id (CDM + Sentinel-2 test sites)')
    for field, title, unit, steps in (
        ('best_model', 'Best model (macro-F1)', 'none', [(None, 'purple')]),
        ('best_f1_macro', 'Best macro-F1', 'percentunit', [(None, 'red'), (0.8, 'orange'),
                                                            (0.9, 'green')]),
        ('n_test', 'Test samples', 'short', [(None, 'blue')]),
        ('n_samples_mapped', 'Samples on risk map', 'short', [(None, 'blue')]),
    ):
        b.add(stat(title, offline('pipeline_run', [field], 'last()'), unit=unit, steps=steps,
                   decimals=1 if unit == 'percentunit' else None, no_value='run pipeline',
                   color_mode='value', fields='/.*/'), 6, 4)
    b.add(barchart('Test accuracy and macro-F1 by model', offline(
        'model_evaluation', ['accuracy', 'f1_macro'],
        'pivot(rowKey: ["model"], columnKey: ["_field"], valueColumn: "_value")', 'group()',
        'keep(columns: ["model", "accuracy", "f1_macro"])'), 'model', unit='percentunit',
        min_=0.8, max_=1,
        overrides=[rename('accuracy', 'Accuracy'), rename('f1_macro', 'Macro-F1')]), 12, 10)
    b.add(table('Pairwise McNemar tests', offline(
        'model_significance', ['p_value', 'only_a_correct', 'only_b_correct', 'significant'],
        'pivot(rowKey: ["model_a", "model_b"], columnKey: ["_field"], valueColumn: "_value")',
        'group()', 'keep(columns: ["model_a", "model_b", "only_a_correct", "only_b_correct", '
        '"p_value", "significant"])'),
        overrides=[rename('model_a', 'Model A'), rename('model_b', 'Model B'),
                   rename('only_a_correct', 'Only A right', prop('decimals', 0)),
                   rename('only_b_correct', 'Only B right', prop('decimals', 0)),
                   rename('p_value', 'p-value', prop('decimals', 4)),
                   rename('significant', 'Significant (p<0.05)', prop(
                       'custom.cellOptions', {'type': 'color-background'}),
                       prop('mappings', [{'type': 'value', 'options': {
                           'true': {'text': 'yes', 'color': 'green', 'index': 0},
                           'false': {'text': 'no', 'color': 'transparent', 'index': 1}}}]))],
        sort_by=[{'displayName': 'p-value', 'desc': False}]), 12, 10)
    comparison = b.add(table('Model comparison with 95% bootstrap CIs', offline(
        'model_evaluation', ['accuracy', 'accuracy_ci_low', 'accuracy_ci_high', 'f1_macro',
                             'f1_macro_ci_low', 'f1_macro_ci_high', 'f1_early_stress'],
        'pivot(rowKey: ["model"], columnKey: ["_field"], valueColumn: "_value")', 'group()',
        'keep(columns: ["model", "accuracy", "accuracy_ci_low", "accuracy_ci_high", '
        '"f1_macro", "f1_macro_ci_low", "f1_macro_ci_high", "f1_early_stress"])'),
        unit='percentunit',
        overrides=[rename('model', 'Model'), rename('accuracy', 'Accuracy'),
                   rename('accuracy_ci_low', 'Acc CI low'),
                   rename('accuracy_ci_high', 'Acc CI high'),
                   rename('f1_macro', 'Macro-F1', prop('custom.cellOptions', {
                       'type': 'color-background', 'mode': 'gradient'}),
                       prop('color', {'mode': 'continuous-RdYlGr'}), prop('min', 0.85),
                       prop('max', 1)),
                   rename('f1_macro_ci_low', 'F1 CI low'),
                   rename('f1_macro_ci_high', 'F1 CI high'),
                   rename('f1_early_stress', 'Early-Stress F1')],
        sort_by=[{'displayName': 'Macro-F1', 'desc': True}]), 24, 8)
    comparison['transformations'] = [{'id': 'organize', 'options': {'indexByName': {
        'model': 0, 'accuracy': 1, 'accuracy_ci_low': 2, 'accuracy_ci_high': 3,
        'f1_macro': 4, 'f1_macro_ci_low': 5, 'f1_macro_ci_high': 6, 'f1_early_stress': 7}}}]
    b.add(trend('Training loss by epoch',
                epoch_curves({'train': 'train_loss', 'validation': 'val_loss'}), 'epoch',
                description='Per-epoch training and validation loss of each sequence '
                            'model'), 12, 8)
    b.add(trend('Validation macro-F1 by epoch',
                epoch_curves({'val macro-F1': 'val_macro_f1', 'train accuracy': 'train_acc'}),
                'epoch', unit='percentunit',
                description='Checkpoints are chosen at the epoch with the best validation '
                            'macro-F1'), 12, 8)
    b.add(barchart('Classical CV: random vs spatial-zone hold-out (macro-F1)', offline(
        'cross_validation', ['cv_f1_macro_mean'],
        'pivot(rowKey: ["model"], columnKey: ["cv_split"], valueColumn: "_value")', 'group()',
        'keep(columns: ["model", "random", "spatial_zone"])'), 'model', unit='percentunit',
        min_=0.9, max_=1,
        overrides=[rename('random', 'Random 5-fold'),
                   rename('spatial_zone', 'Spatial zone hold-out')]), 8, 9)
    b.add(barchart('Feature selection: QUBO vs baselines (macro-F1)', offline(
        'feature_selection', ['cv_f1_macro_mean', 'test_f1_macro'],
        'pivot(rowKey: ["method"], columnKey: ["_field"], valueColumn: "_value")', 'group()',
        'keep(columns: ["method", "cv_f1_macro_mean", "test_f1_macro"])'), 'method',
        unit='percentunit', min_=0.8, max_=1,
        overrides=[rename('cv_f1_macro_mean', 'CV macro-F1'),
                   rename('test_f1_macro', 'Test macro-F1')]), 16, 9)
    b.add(geomap('Batch risk map — every sample, best model', offline(
        'batch_risk', ['latitude', 'longitude', 'risk_score', 'predicted_label', 'true_label',
                       'sample_id'],
        'pivot(rowKey: ["_time", "split"], columnKey: ["_field"], valueColumn: "_value")',
        'group()'), 'risk_score', size=5, zoom=8,
        description='Pipeline predictions for all samples (train/val/test) at the real '
                    'CDM report sites. Labels are time-relative, so a site colour shows '
                    'where its sequences fall in the outbreak timeline.'), 12, 9)
    b.add(timeseries('Macro-F1 across pipeline runs', flux(
        'model_evaluation', ['f1_macro'], False, '  |> range(start: -365d)',
        'keep(columns: ["_time", "_value", "model"])'),
        unit='percentunit', display_name='${__field.labels.model}', points='always',
        fill=0, description='One point per model per pipeline run (last 365 days, '
                            'independent of the dashboard time range)'), 12, 9)
    return b


def system() -> Board:
    b = Board('crop-system', 'Crop Disease — System Health',
              'Data pipeline health: ingest rate per station, data freshness, bridge '
              'counters and payload errors.', LIVE_VARS, time_from='now-3h', refresh='10s')

    stats = flux('bridge_stats', [], False, '  |> range(start: -24h)')
    b.row('Now')
    b.add(stat('Seconds since last reading', flux(
        'spectral_readings', ['nir_842'], True, '  |> range(start: -30d)', 'group()', 'last()',
        'map(fn: (r) => ({_value: float(v: int(v: now()) - int(v: r._time)) / 1000000000.0}))'),
        unit='s', decimals=0, no_value='no data',
        steps=[(None, 'green'), (60, 'orange'), (300, 'red')],
        description='Age of the most recent reading from the selected stations'), 6, 4)
    b.add(stat('Readings per minute (last 5 min)', flux(
        'spectral_readings', ['nir_842'], True, '  |> range(start: -5m)', 'group()',
        'count()', 'map(fn: (r) => ({_value: float(v: r._value) / 5.0}))'),
        unit='short', decimals=1, steps=[(None, 'red'), (0.5, 'green')]), 6, 4)
    for field, title, steps in (
        ('invalid', 'Invalid payloads dropped', [(None, 'green'), (1, 'orange')]),
        ('write_errors', 'InfluxDB write errors', [(None, 'green'), (1, 'red')]),
    ):
        b.add(stat(title, stats.replace(
            '  |> filter(fn: (r) => r._measurement == "bridge_stats")',
            '  |> filter(fn: (r) => r._measurement == "bridge_stats" and r._field == "'
            + field + '")') + '\n  |> last()', steps=steps, no_value='bridge offline',
            description='Since the bridge last started'), 6, 4)

    b.row('Throughput')
    b.add(timeseries('Readings per minute by station', flux(
        'spectral_readings', ['nir_842'], True, RANGE,
        'aggregateWindow(every: 1m, fn: count, createEmpty: true)',
        'keep(columns: ["_time", "_value", "station_id"])'),
        unit='short', display_name='${__field.labels.station_id}', draw='bars', fill=70,
        stack=True, legend_calcs=('mean', 'max')), 16, 8)
    b.add(piechart('Data by source', flux(
        'spectral_readings', ['nir_842'], True, RANGE, 'group(columns: ["source"])', 'count()',
        'keep(columns: ["_value", "source"])'), display_name='${__field.labels.source}',
        description='simulator = live replay over MQTT, backfill = seed_history'), 8, 8)
    b.add(state_timeline('Station heartbeat (readings per 5 min)', flux(
        'spectral_readings', ['nir_842'], True, RANGE,
        'aggregateWindow(every: 5m, fn: count, createEmpty: true)',
        'map(fn: (r) => ({r with _value: if r._value > 0 then "reporting" else "silent"}))',
        'keep(columns: ["_time", "_value", "station_id"])'),
        states={'reporting': 'green', 'silent': 'red'},
        description='Green while a station delivered readings in each 5-minute window'),
        24, 7)

    b.row('MQTT → InfluxDB bridge')
    b.add(timeseries('Bridge throughput (per minute)', stats.replace(
        '  |> filter(fn: (r) => r._measurement == "bridge_stats")',
        '  |> filter(fn: (r) => r._measurement == "bridge_stats" and '
        '(r._field == "processed" or r._field == "predictions"))') +
        '\n  |> derivative(unit: 1m, nonNegative: true)'
        '\n  |> keep(columns: ["_time", "_value", "_field"])',
        unit='short', fill=15, description='Rate of readings processed and predictions made '
                                           '(from the bridge\'s cumulative counters)'), 12, 8)
    b.add(timeseries('Bridge errors (cumulative since start)', stats.replace(
        '  |> filter(fn: (r) => r._measurement == "bridge_stats")',
        '  |> filter(fn: (r) => r._measurement == "bridge_stats" and '
        '(r._field == "invalid" or r._field == "write_errors"))') +
        '\n  |> keep(columns: ["_time", "_value", "_field"])',
        unit='short', fill=0, steps=[(None, 'green'), (1, 'red')]), 12, 8)
    b.add(table('Station freshness', flux(
        'spectral_readings', ['nir_842'], True, '  |> range(start: -30d)',
        'group(columns: ["station_id", "source"])', 'last()',
        'map(fn: (r) => ({station_id: r.station_id, source: r.source, last_reading: r._time, '
        'age_s: float(v: int(v: now()) - int(v: r._time)) / 1000000000.0}))', 'group()'),
        overrides=[rename('station_id', 'Station'), rename('source', 'Source'),
                   rename('last_reading', 'Last reading'),
                   rename('age_s', 'Age', prop('unit', 's'), prop('decimals', 0),
                          prop('custom.cellOptions', {'type': 'color-background'}),
                          prop('thresholds', thresholds([(None, 'green'), (60, 'orange'),
                                                         (300, 'red')])),
                          prop('color', {'mode': 'thresholds'}))],
        sort_by=[{'displayName': 'Age', 'desc': True}]), 24, 7)
    return b


BOARDS = {'crop_overview.json': overview, 'crop_spectral.json': spectral,
          'crop_models.json': models, 'crop_system.json': system}


def render() -> dict[str, str]:
    return {name: json.dumps(build().to_json(), indent=2, ensure_ascii=False) + '\n'
            for name, build in BOARDS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--check', action='store_true',
                        help='Exit 1 if the committed JSON differs from the generator output')
    args = parser.parse_args()
    stale = []
    for name, content in render().items():
        path = os.path.join(OUT_DIR, name)
        current = open(path, encoding='utf-8').read() if os.path.exists(path) else None
        if args.check:
            if current != content:
                stale.append(name)
        elif current != content:
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            print(f'wrote {path}')
    if stale:
        print(f'Stale dashboards (run without --check): {stale}')
        sys.exit(1)


if __name__ == '__main__':
    main()
