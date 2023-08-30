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

function myFunction() {
  
  // TODO (developer) - Replace this value with the project ID listed in the Google
  // Cloud Platform project.
  const projectId = 'google.com:sight-226816';

  const request = {
    // TODO (developer) - Replace query with yours
    query: 'SELECT *' +
      'FROM `test.schemaReference`;',
    useLegacySql: false
  };
  let queryResults = BigQuery.Jobs.query(request, projectId);
  const jobId = queryResults.jobReference.jobId;

  // Check on status of the Query Job.
  let sleepTimeMs = 500;
  while (!queryResults.jobComplete) {
    Utilities.sleep(sleepTimeMs);
    sleepTimeMs *= 2;
    queryResults = BigQuery.Jobs.getQueryResults(projectId, jobId);
  }

  // Get all the rows of results.
  let rows = queryResults.rows;
  while (queryResults.pageToken) {
    queryResults = BigQuery.Jobs.getQueryResults(projectId, jobId, {
      pageToken: queryResults.pageToken
    });
    rows = rows.concat(queryResults.rows);
  }

  if (!rows) {
    Logger.log('No rows returned.');
    return;
  }
  const spreadsheet = SpreadsheetApp.create('schemaReference Results from BQ table');
  const sheet = spreadsheet.getActiveSheet();

  // Append the headers.
  const headers = queryResults.schema.fields.map(function(field) {
    return field.name;
  });
  sheet.appendRow(headers);

  // Append the results.
  var data = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const cols = rows[i].f;
    data[i] = new Array(cols.length);
    for (let j = 0; j < cols.length; j++) {
      data[i][j] = cols[j].v;
    }
  }
  sheet.getRange(2, 1, rows.length, headers.length).setValues(data);

  Logger.log('Results spreadsheet created: %s',
      spreadsheet.getUrl());
}

