// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var sightService = 'https://bronevet-sight.sandbox.googleapis.com/v1';
var sightApiKey = '<API key here';

function doGet(e) {
  var template = HtmlService
    .createTemplateFromFile('Index');
  template.log_id = e.parameters['log_id'];
  template.log_owner = e.parameters['log_owner'];
  template.selections = e.parameters['selections'];
  return template.evaluate();
}

function include(filename) {
  return HtmlService.createHtmlOutputFromFile(filename)
    .setFaviconUrl('https://git.teknik.io/mushmouth/Website/raw/commit/82740c80442d871a5e2bbbb83d25f6a060b6ad68/favicon.ico')
    .getContent();
}

function executeProjection(query) {
  // var request = {
  //   queryRequest: {
  //     query: {
  //       text: query,
  //       engine: "BIG_QUERY",
  //       dialect: "Standard SQL"
  //     }
  //   }
  // };
  const projectId = 'cameltrain';
  var request = {
    query: query,
    useLegacySql: false
  };
  console.log('query=' + query);
  try {
    // ---Old Partially Working Code
    // // var projection = Plx.Projections.create(request);
    // let projection = BigQuery.Jobs.query(request, projectId);
    // const jobId = projection.jobReference.jobId;
    // // while (projection.state !== 'done') {
    // //   Utilities.sleep(1000);
    // //   projection = Plx.Projections.get(projection.id, { token: projection.token });
    // // }
    // var sleepTimeMs = 500;
    // var rows = projection.rows;

    // console.log("Savan rows before:",projection.totalRows, projection.jobComplete);
    // while (!projection.jobComplete) {
    //   console.log("In my loop");
    //   Utilities.sleep(sleepTimeMs);
    //   sleepTimeMs *= 2;
    //   // projection = BigQuery.Jobs.getprojection(projectId, jobId);
    //   projection = BigQuery.Jobs.get(projectId, jobId);
    //   rows.push(projection.rows);
    // }
    
    // New Tries --
    var queryResults = BigQuery.Jobs.query(request, projectId);
    var jobId = queryResults.jobReference.jobId;
    queryResults = BigQuery.Jobs.getQueryResults(projectId, jobId);

    var rows = queryResults.rows;
    console.log("Meet rows before:",rows.length);
    console.log("before - totalRows:",queryResults.totalRows);
    console.log("before - jobComplete:", queryResults.jobComplete);
    console.log("before - pageToken:", queryResults.pageToken);

    while (queryResults.pageToken) {
      queryResults = BigQuery.Jobs.getQueryResults(projectId, jobId, {
        pageToken: queryResults.pageToken
      });
      rows = rows.concat(queryResults.rows);
      console.log("In between row count is : ",rows.length, queryResults.totalBytesProcessed);
    }

    console.log("Meet rows after:",rows.length);
    console.log("after - totalRows:",queryResults.totalRows);
    console.log("after - jobComplete:", queryResults.jobComplete);
    console.log("after - pageToken:", queryResults.pageToken);


    //New Tries --
    // var max = 10000;
    
    // var queryResults = BigQuery.Jobs.query(request, projectId, {
    //     maxResults: max
    //   });
  
    // var jobId = queryResults.jobReference.jobId;
    // queryResults = BigQuery.Jobs.getQueryResults(projectId, jobId, {
    //     maxResults: max
    //   });
  
    // var rows = queryResults.rows;
  
    // let i = 0;
    // while (i < 17) {
    //   queryResults = BigQuery.Jobs.getQueryResults(projectId, jobId, {
    //     pageToken: queryResults.pageToken,
    //     maxResults: max
    //   });
    //     i++;
    //   rows = rows.concat(queryResults.rows); 
    //   console.log("In between row count is : ",rows.length);
    // }
    
    // console.log(rows.length);

    return {
      'success': true,
      'projection': queryResults,
      'rows': rows
    };
  } catch (e) {
    Logger.log(e);
    return {
      'success': false,
      'err_message': e,
    };
  }
}

function logQueryOutputToHash(response) {
  var rawData = response.log.obj;
  Logger.log('rawData');
  Logger.log(rawData);
  var data = [];
  for (var i = 0; i < rawData.length; ++i) {
    rawData[i].obj['selected'] = response.selected[i];
    Logger.log(rawData[i]);
    data.push(rawData[i].obj);
  }
  Logger.log('data');
  Logger.log(data);
  return data;

  var table = Utilities.parseCsv(rawData);

  // data.replace(/(["'])(?:(?=(\\?))\2[\s\S])*?\1/g, function(e){return e.replace(/\r?\n|\r/g, '\\n') }));
  Logger.log('table');
  Logger.log(table);
  var records = [];

  Logger.log("table.length=" + table.length);
  //  Logger.log(table[0]);
  for (var row = 1; row < table.length; row++) {
    var r = {};
    Logger.log(table[row]);
    for (var col = 0; col < table[0].length; col++) {
      r[table[0][col].replace(/^log[.]/, '')] = table[row][col];
    }
    if (r["location"] == "") {
      continue;
    }
    if (r["sub_type"] == "ST_ATTRIBUTE_START" ||
      r["sub_type"] == "ST_ATTRIBUTE_END") {
      continue;
    }
    var attrVal = r["attribute.value"];

    // If this is a second record for the same location that communicates a different
    // attribute value.
    if (records.length > 0 && records[records.length - 1]["location"] == r["location"]) {
      if (r["attribute.key"] && r["attribute.key"] != "null") {
        if (records[records.length - 1]["attributes"] == undefined) {
          records[records.length - 1]["attributes"] = {};
        }
        records[records.length - 1]["attributes"][r["attribute.key"]] = attrVal;
      }
    } else {
      if (r["attribute.key"] && r["attribute.key"] != "null") {
        r["attributes"] = {};
        r["attributes"][r["attribute.key"]] = attrVal;
      }
      delete r["attribute.key"];
      delete r["attribute.value"];
      records.push(r);
    }
  }
  Logger.log('records');
  Logger.log(records);
  return records;
}

function queryOutputToHash(data) {
  console.log(" In queryOutputToHash...... ");
  // Logger.log(data);

  var fields = data.projection.schema.fields
  fieldNames = []
  for (var field of fields) {
    fieldNames.push(field.name)
  }
  console.log("fieldNames : ", fieldNames)

  var records = [];
  var rows = data.projection.rows
  for (var row of rows) {
    // console.log('row : ',row)
    var i = row['f']
    var r = {};
    for (var col = 0; col < fieldNames.length; col++) {
      r[fieldNames[col]] = i[col].v
    }
    records.push(r);
    // }
  }
  return records;
  // console.log(" In queryOutputToHash...... ");
  // // Logger.log(data);
  // // let sanitizedData =
  // //       data.replace(/\\/g, '::back-slash::')
  // //           .replace(/(?=["'])(?:"[^"\\]*(?:\\[\s\S][^"\\]*)*"|'[^'\\]\r?\n(?:\\[\s\S][^'\\]\r?\n)*')/g,
  // //               match => match.replace(/\r?\n/g, "::newline::"));
  // var table = Utilities.parseCsv(data); //sanitizedData);
  // var records = [];
  // for (var row = 1; row < table.length; row++) {
  //   //    Logger.log(table[row]);
  //   var r = {};
  //   for (var col = 0; col < table[0].length; col++) {
  //     r[table[0][col]] = table[row][col];
  //   }
  //   records.push(r);
  // }
  // return records;
}

//function initializeLogView(logId) {
//  var logQuery =
//'SELECT\n\
//  *\n\
//FROM\n\
//  bronevet.sight.test_log\n\
//ORDER BY\n\
//  index';
//  
//  var logDataHash = queryOutputToHash(executeProjection(logQuery).data);
//  
//  return {
//    'logData': logDataHash,
//    'logQuery': logQuery,
//  };
//}

function getAttributes(logId, logOwner) {
  console.log('logOwner=' + logOwner);
  var tablePrefix = logOwner.replace(/-/g, '_').replace(/^mdb[/]/, '').replace(/@google.com$/, '');
  console.log('tablePrefix=' + tablePrefix);
  var options = {
    'method': 'post',
    'payload': {
      'id': logId,
      'log_owner': logOwner,
      'table_prefix': tablePrefix,
    }
    // muteHttpExceptions: true
  };
  // Logger.log(options);
  var response = UrlFetchApp.fetch(sightService + '/getAttributes?key=' + sightApiKey, options);
  // Logger.log(response);
  // Logger.log(JSON.parse(response.getContentText()));
  return JSON.parse(response.getContentText());
}

function initializeLogView(logId, logOwner) {
  // var logTable = logOwner.replace(/-/g, '_').replace(/^mdb[/]/, '').replace(/@google.com$/, '')
  //                +'.sight.'+logId;
  // var logTable = logOwner.replace(/-/g, '_').replace(/^mdb[/]/, '').replace(/@google.com$/, '')
  //                +'.'+logId;
  var logTable = logId;
  console.log('logTable=' + logTable)
  var attributesQuery = 'SELECT a.key AS key, a.value AS value FROM ' + logTable + ', UNNEST(attribute) as a WHERE a.key IS NOT NULL GROUP BY key, value ORDER BY key;';
  var result = executeProjection(attributesQuery);
  console.log('result=' + JSON.stringify(result))
  if (!result.success) {
    return result;
  }
  console.log('result.projection.rows=' + result.projection.rows)
  var attributesDataHash = queryOutputToHash(result);
  console.log('attributesDataHash=', attributesDataHash)

  var attributesValueListsHash = {};
  for (var entry of attributesDataHash) {
    if (attributesValueListsHash.hasOwnProperty(entry.key)) {
      attributesValueListsHash[entry.key].push(entry.value)
    } else {
      attributesValueListsHash[entry.key] = [entry.value]
    }
  }

  // var attributesValueListsHash = {};
  // for(var entry of result.projection.rows){
  //   var i = entry['f']
  //   var key = i[0].v
  //   var values = i[1].v
  //   // console.log("key : ", key )
  //   // console.log("values : ", values )

  //   for(var val of values){
  //     if (attributesValueListsHash.hasOwnProperty(key)) {
  //       attributesValueListsHash[key].push(val.v)
  //     } else {
  //       attributesValueListsHash[key] = [val.v]
  //     }
  //   }
  // }

  console.log('attributesValueListsHash');
  console.log(attributesValueListsHash);

  var attributesValueListsList = []
  for (var key in attributesValueListsHash) {
    attributesValueListsList.push({
      key: key,
      value: attributesValueListsHash[key]
    })
  }
  console.log('attributesValueListsList');
  console.log(attributesValueListsList);

  return {
    'success': true,
    'attributesData': attributesValueListsList,
    //getAttributes(logId, logOwner)['attributeValues'],
    'attributesQuery': '',
  };
}

function withQuery(label, query, suffix, indent) {
  return indent + label + ' AS\n' +
    indent + '(\n' +
    query + '\n' +
    indent + ')' + suffix;
}


function locsAndAncestors(logTable, includeHierarchicalContext, suffix, indent) {
  var query = indent + 'SELECT\n' +
    indent + '  location,';
  if (includeHierarchicalContext) {
    query += '\n' + indent + '  ancestor_start_location\n';
  } else {
    query += '\n' + indent + '  location as ancestor_start_location\n';
  }
  query += '\n' + indent + 'FROM\n' +
    indent + '  ' + logTable + ' AS log';
  if (includeHierarchicalContext) {
    query += indent + '\nCROSS JOIN UNNEST(log.ancestor_start_location) as ancestor_start_location\n';
  }
  return withQuery('locsAndAncestors', query, suffix, indent);
}

function logAllCols(logTable, suffix, indent) {
  return withQuery('logAllCols',
    indent + 'SELECT\n' +
    indent + '  IF(subType = "ST_BLOCK_END", block_end_location_of_block_start, location) AS anchor_loc,\n' +
    indent + '  *,\n' +
    // indent + '  "" AS parent_location,'+
    indent + '  IF(ARRAY_LENGTH(SPLIT(ancestor_start_location,","))=1, "",\n' +
    indent + '    SPLIT(ancestor_start_location,",")[OFFSET(ARRAY_LENGTH(SPLIT(ancestor_start_location,","))-2)]) AS parent_location\n' +
    indent + 'FROM\n' +
    indent + '(\n' +
    indent + '  SELECT\n' +
    // indent + '    AS sight.x.proto.Object *\n' +
    indent + '    location,\n' +
    indent + '    index,\n' +
    indent + '    sub_type AS subType,\n' +
    indent + '    ARRAY_TO_STRING(ancestor_start_location, ",") AS ancestor_start_location,\n' +
    indent + '    TO_JSON_STRING(text) AS text,\n' +
    indent + '    TO_JSON_STRING(tensor) AS tensor,\n' +
    indent + '    TO_JSON_STRING(flume_do_fn_emit) AS flumeDoFnEmit,\n' +
    indent + '    TO_JSON_STRING(flume_depend) AS flumeDepend,\n' +
    indent + '    TO_JSON_STRING(block_start) AS blockStart,\n' +
    indent + '    TO_JSON_STRING(block_end) AS blockEnd,\n' +
    indent + '    block_end.location_of_block_start as block_end_location_of_block_start,\n' +
    indent + '    TO_JSON_STRING(value) AS value,\n' +
    indent + '    TO_JSON_STRING(exception) AS exception,\n' +
    indent + '    file,\n' +
    indent + '    line,\n' +
    indent + '    func\n' +
    indent + '  FROM\n' +
    indent + '    ' + logTable + '\n' +
    indent + ') AS obj', suffix, indent);
}

function attr(logTable, attrIdx, keyName, keyVal, relOp, suffix, indent) {
  return withQuery('attr_' + attrIdx,
    indent + 'SELECT\n' +
    indent + '    location\n' +
    indent + 'FROM\n' +
    indent + '(\n' +
    indent + '  SELECT\n' +
    indent + '    location,\n' +
    indent + '    attribute\n' +
    indent + '  FROM ' + logTable + ' AS log\n' +
    indent + '  CROSS JOIN UNNEST(log.attribute) as attribute\n' +
    indent + ')\n' +
    indent + 'WHERE\n' +
    indent + '  attribute.key="' + keyName.replaceAll('"', '\\"') + '" AND attribute.value' + relOp + '"' + keyVal.replaceAll('"', '\\"') + '"', suffix, indent);
}

function maxDepthQuery(logTable, depth, suffix, indent) {
  if (depth == 0) {
    return '';
  }
  return withQuery('maxDepth',
    indent + 'SELECT\n' +
    indent + '  location\n' +
    indent + 'FROM\n' +
    indent + '(\n' +
    indent + '  SELECT\n' +
    indent + '    location,\n' +
    indent + '    ARRAY_LENGTH(SPLIT(location, ":")) AS depth\n' +
    indent + '  FROM ' + logTable + '\n' +
    indent + ') WHERE depth <= ' + depth + '\n',
    suffix, indent);
}

function logRangeQuery(logTable, rangeStartLocation, rangeEndLocation, suffix, indent) {
  if (rangeStartLocation == '' && rangeEndLocation == '') {
    return '';
  }
  query =
    indent + 'SELECT\n' +
    indent + '  location\n' +
    indent + 'FROM ' + logTable + '\n' +
    indent + 'WHERE\n';
  if (rangeStartLocation != '') {
    query +=
      indent + '  location >= "' + rangeStartLocation + '"\n';
  }
  if (rangeStartLocation != '') {
    query += '  AND\n';
  }
  if (rangeEndLocation != '') {
    query +=
      indent + '  location <= "' + rangeEndLocation + '"\n';
  }
  return withQuery('logRange', query, suffix, indent);
}

function matches(logTable, numAttrs, selection, includeMaxDepth, includeRange, suffix, indent) {
  var query = '';
  query +=
    indent + 'SELECT * FROM (\n' +
    indent + '  SELECT\n' +
    indent + '    locsAndAncestors.ancestor_start_location AS ancestor,\n' +
    indent + '  FROM\n' +
    indent + '    locsAndAncestors\n';
  for (var attrIdx = 0; attrIdx < numAttrs; attrIdx++) {
    query += indent + '  JOIN attr_' + attrIdx + '\n' +
      indent + '  ON locsAndAncestors.location = attr_' + attrIdx + '.location\n';
  }
  if (includeMaxDepth) {
    query += indent + '  JOIN maxDepth ON locsAndAncestors.location = maxDepth.location\n';
  }
  if (includeRange) {
    query += indent + '  JOIN logRange ON locsAndAncestors.location = logRange.location\n';
  }
  query += indent + '  GROUP BY ancestor' +
    indent + ')';
  return withQuery('matches', query, suffix, indent);
}

function selectedObjects(logTable, suffix, indent) {
  return withQuery('selectedObjects',
    indent + 'SELECT\n' +
    indent + '  "" AS selected,\n' +
    indent + '  log.*,\n' +
    // indent + '  log.anchor_loc,\n' +
    // indent + '  log.obj,\n' +
    // indent + '  log.parent_location\n' +
    indent + 'FROM\n' +
    indent + '  matches\n' +
    indent + 'JOIN\n' +
    indent + '  logAllCols as log\n' +
    indent + 'ON\n' +
    indent + '  matches.ancestor = log.anchor_loc', suffix, indent);
}

function notSelected(logTable, suffix, indent) {
  return withQuery('notSelected',
    indent + 'SELECT\n' +
    indent + '  *\n' +
    indent + '  FROM\n' +
    indent + '  (\n' +
    indent + '    SELECT\n' +
    indent + '      selectedObjects.location AS selected,\n' +
    indent + '      logAllCols.*,\n' +
    // indent + '      logAllCols.anchor_loc,\n' +
    // indent + '      logAllCols.obj,\n' +
    // indent + '      logAllCols.parent_location,\n' +
    indent + '    FROM\n' +
    indent + '      logAllCols\n' +
    indent + '    LEFT JOIN\n' +
    indent + '      selectedObjects\n' +
    indent + '    ON\n' +
    indent + '      logAllCols.location = selectedObjects.location\n' +
    indent + '  )\n' +
    indent + '  WHERE\n' +
    indent + '    selected IS NULL', suffix, indent);
}

function rootsOfUnselectedLogObjectsWithinSelection(logTable, suffix, indent) {
  return withQuery('rootsOfUnselectedLogObjectsWithinSelection',
    indent + 'SELECT\n' +
    indent + '  notSelected.*\n' +
    indent + 'FROM\n' +
    indent + '  notSelected\n' +
    indent + 'JOIN\n' +
    indent + '  selectedObjects\n' +
    indent + 'ON\n' +
    indent + '  notSelected.parent_location = selectedObjects.location', suffix, indent);
}

function rootsOfUnselectedLogObjectsAtTopLevel(logTable, suffix, indent) {
  return withQuery('rootsOfUnselectedLogObjectsAtTopLevel',
    indent + 'SELECT\n' +
    indent + '  *\n' +
    indent + 'FROM\n' +
    indent + '  notSelected\n' +
    indent + 'WHERE\n' +
    indent + '  parent_location = ""', suffix, indent);
}

function allObjects(logTable, includeHierarchicalContext, suffix, indent) {
  query =
    indent + 'SELECT * FROM\n' +
    indent + '  (\n' +
    indent + '    (SELECT * FROM selectedObjects)\n' +
    indent + '  UNION ALL\n' +
    indent + '    (SELECT * FROM rootsOfUnselectedLogObjectsWithinSelection)\n';
  if (includeHierarchicalContext) {
    query += '  UNION ALL\n' +
      indent + '    (SELECT * FROM rootsOfUnselectedLogObjectsAtTopLevel)\n';
  }
  query += indent + ')';
  return withQuery('allObjects', query, suffix, indent);
}

function allObjectsWithoutAdjacentUnselectedLogObjects(logTable, suffix, indent) {
  return withQuery('allObjectsWithoutAdjacentUnselectedLogObjects',
    indent + 'SELECT\n' +
    indent + '  *\n' +
    indent + 'FROM\n' +
    indent + '(\n' +
    indent + '  SELECT\n' +
    indent + '    NTH_VALUE(parent_location, 2) OVER earlier_neighbors AS last_parent_location,\n' +
    indent + '    NTH_VALUE(selected, 2) OVER earlier_neighbors AS last_selected,\n' +
    indent + '    NTH_VALUE(parent_location, 2) OVER later_neighbors AS next_parent_location,\n' +
    indent + '    NTH_VALUE(selected, 2) OVER later_neighbors AS next_selected,\n' +
    indent + '    *\n' +
    indent + '  FROM\n' +
    indent + '    allObjects\n' +
    indent + '  WINDOW later_neighbors AS (\n' +
    indent + '      ORDER BY location\n' +
    indent + '      RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING),\n' +
    indent + '   earlier_neighbors AS (\n' +
    indent + '      ORDER BY location\n' +
    indent + '      RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)\n' +
    indent + ')\n' +
    indent + 'WHERE\n' +
    indent + '  NOT(\n' +
    indent + '    selected IS NULL AND\n' +
    indent + '    last_parent_location IS NOT NULL AND last_parent_location = parent_location AND last_selected IS NULL AND\n' +
    indent + '    next_parent_location IS NOT NULL AND next_parent_location = parent_location AND next_selected IS NULL)',
    suffix, indent
  );
}

// function convertToNull(hash, field) {
//   if(hash[field] == null)
//   {
//     // console.log("It worked for  : "+field);
//     hash[field] = 'null';
//   }
// }

function parseFromJsonOrNull(hash, field) {
  // console.log(hash);
  // console.log('hash['+field+']='+hash[field]);
  if (hash[field] == 'null' || hash[field] == undefined) {
    hash[field] = undefined;
  } else {
    try {
      hash[field] = JSON.parse(hash[field]);
    } catch (error) {
      console.error(error);
    }
  }
}

function selectionQueryWithAncestorsInObjects2(
  selection, maxDepth, rangeStartLocation, rangeEndLocation, includeHierarchicalContext, delayedLoading, logId, logOwner) {
  console.log("In function selectionQueryWithAncestorsInObjects2");
  var keys = Object.keys(selection);
  // var logTable = logOwner.replace(/-/g, '_').replace(/^mdb[/]/, '').replace(/@google.com$/, '')
  //   + '.sight.' + logId;
  var logTable = logId
  var indent = '';
  // query = 'SET SQL_DIALECT GoogleSQL;\n' +
  // 'LOAD "globaldb:sight.x.proto.Object";\n' +
  query = 'WITH\n' +
    locsAndAncestors(logTable, includeHierarchicalContext, ',\n', indent) +
    logAllCols(logTable, ',\n', indent);
  var numAttrs = keys.length;
  for (var attrIdx = 0; attrIdx < keys.length; attrIdx++) {
    query += attr(logTable, attrIdx, keys[attrIdx], selection[keys[attrIdx]], '=', ',\n', indent);
  }
  if (delayedLoading) {
    query += attr(logTable, keys.length, 'tensor_flow_model_epoch_body', 'false', '=', ',\n', indent);
    ++numAttrs;
  }

  query += maxDepthQuery(logTable, maxDepth, ',\n', indent) +
    logRangeQuery(logTable, rangeStartLocation, rangeEndLocation, ',\n', indent) +
    matches(
      logTable,
      numAttrs,
      selection,
      /* includeMaxDepth= */ maxDepth > 0,
      /* includeRange= */ rangeStartLocation != '' || rangeEndLocation != '',
      ',\n', indent) +
    selectedObjects(logTable, ',\n', indent) +
    notSelected(logTable, ',\n', indent) +
    rootsOfUnselectedLogObjectsWithinSelection(logTable, ',\n', indent) +
    rootsOfUnselectedLogObjectsAtTopLevel(logTable, ',\n', indent) +
    allObjects(logTable, includeHierarchicalContext, ',\n', indent) +
    allObjectsWithoutAdjacentUnselectedLogObjects(logTable, '\n', indent) +
    'SELECT * FROM allObjectsWithoutAdjacentUnselectedLogObjects';

  console.log(" query executing : ");

  var result = executeProjection(query);
  if (!result.success) {
    console.log(" query not executed correctly here: ");
    return result;
  }

  console.log("result : ", JSON.stringify(result))

  // console.log("result : ", JSON.stringify(result))

  var logDataHash = queryOutputToHash(result);
  // console.log('logDataHash : ', logDataHash)
  // console.log("check till here")
  console.log("Before parseFromJsonOrNull logDataHash : ", logDataHash)
  // console.log(logDataHash);
  for (var row of logDataHash) {
    ;
    parseFromJsonOrNull(row, 'text');
    parseFromJsonOrNull(row, 'tensor');
    parseFromJsonOrNull(row, 'flumeDoFnEmit');
    parseFromJsonOrNull(row, 'flumeDepend');
    parseFromJsonOrNull(row, 'blockStart');
    parseFromJsonOrNull(row, 'blockEnd');
    parseFromJsonOrNull(row, 'value');
    parseFromJsonOrNull(row, 'exception');

    // convertToNull(row, 'last_parent_location');
    // convertToNull(row, 'last_selected');
    // convertToNull(row, 'block_end_location_of_block_start');

    row['selected'] = (row['selected'] != 'null');
    // console.log(row);
  }
  console.log("Before logDataHash : ", logDataHash);
  // console.log("Idhar se  gaya");
  // logDataHash[0].index = '0';
  return {
    'success': true,
    'logData': logDataHash,
  }
}

function selectionQueryWithAncestorsInObjects3(
  selection, maxDepth, rangeStartLocation, rangeEndLocation, includeHierarchicalContext, logId, logOwner) {
  console.log('selection');
  console.log(selection);
  var attribute = [];
  for (var key in selection) {
    attribute.push({
      'key': key,
      'value': selection[key]
    });
  }
  console.log('attribute');
  console.log(attribute);
  var tablePrefix = logOwner.replace(/-/g, '_').replace(/^mdb[/]/, '').replace(/@google.com$/, '');
  // var options = {
  //   'method' : 'post',
  //   'payload' : JSON.stringify({
  //     'id': logId,
  //     'table_prefix': tablePrefix,
  //     'attribute': attribute
  //   }),
  //   // muteHttpExceptions: true
  // };

  var options = {
    'method': 'post',
    'contentType': 'application/json',
    'payload': JSON.stringify({
      'id': logId,
      'table_prefix': tablePrefix,
      'attribute': attribute,
      'max_depth': maxDepth,
      'range_start_location': rangeStartLocation,
      'range_end_location': rangeEndLocation,
      'include_hierarchical_context': includeHierarchicalContext,
    }),
    'muteHttpExceptions': true
  };

  Logger.log(options);
  var response = UrlFetchApp.fetch(sightService + '/searchByAttributes?key=' + sightApiKey, options);
  Logger.log('searchByAttributes response');
  Logger.log(response);
  Logger.log(JSON.parse(response.getContentText()));
  return JSON.parse(response.getContentText());
}

function select(selection, maxDepth, rangeStartLocation, rangeEndLocation, includeHierarchicalContext, delayedLoading, logId, logOwner,
  containerDomEltId, placeholderDomEltId, callback) {
  console.log("In function select....");
  console.log("selection");
  console.log(selection);
  var results = selectionQueryWithAncestorsInObjects2(
    selection, maxDepth, rangeStartLocation, rangeEndLocation, includeHierarchicalContext, delayedLoading, logId, logOwner);
  // console.log("Idhar aaya")
  console.log("result : ", results)
  if (results.success) {
    var success = false;
    var errMessage = "";
    // var logDataHash = {};
    // if (results['success']) {
    //   logDataHash = logQueryOutputToHash(results);
    //   // var logDataHash = logQueryOutputToHash(executeProjection(query).data);
    //   console.log("logDataHash");
    //   console.log(logDataHash);
    // }


    console.log("Out of function select....")
    return {
      'success': true,
      // 'query': query,
      'logData': results['logData'], //logDataHash,
      'containerDomEltId': containerDomEltId,
      'placeholderDomEltId': placeholderDomEltId,
      'callback': callback,
    };
  } else {
    return {
      'success': false,
      'err_message': results.err_message,
      'containerDomEltId': containerDomEltId,
      'placeholderDomEltId': placeholderDomEltId,
    }
    return results;
  }
}

function addUserAccess(logId, logOwner, userName) {
  var options = {
    'method': 'post',
    'payload': {
      'id': logId,
      'log_owner': logOwner,
      'add_user': userName
    }
    // muteHttpExceptions: true
  };
  Logger.log(options);
  var response = UrlFetchApp.fetch(sightService + '/modifyAccess?key=' + sightApiKey, options);
  Logger.log(response);
  Logger.log(JSON.parse(response.getContentText()));
  return JSON.parse(response.getContentText());
}
