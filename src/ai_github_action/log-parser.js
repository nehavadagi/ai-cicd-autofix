/**
 * Parses CI logs to extract Java compilation errors
 */

function parseLogForError(logData) {
  const lines = logData.split('\n');
  let errorBlock = [];
  let inErrorBlock = false;

  // Look for compilation error patterns
  for (const line of lines) {
    // Maven error pattern
    if (line.includes('[ERROR]') && line.includes('.java:')) {
      inErrorBlock = true;
      errorBlock = [line];
    }
    // Gradle error pattern
    else if (line.toLowerCase().includes('error:') && line.includes('.java:')) {
      inErrorBlock = true;
      errorBlock = [line];
    }
    // Continue collecting error lines
    else if (inErrorBlock && (line.includes('[ERROR]') || line.trim().startsWith('at '))) {
      errorBlock.push(line);
    }
    // End of error block
    else if (inErrorBlock && line.trim() === '') {
      break;
    }
  }

  if (errorBlock.length === 0) {
    return null;
  }

  // Extract information from the first error line
  const firstErrorLine = errorBlock[0];
  const fileMatch = firstErrorLine.match(/([a-zA-Z0-9_\/]+\.java):\[?(\d+)/);
  
  if (!fileMatch) {
    return null;
  }

  const filePath = fileMatch[1];
  const lineNumber = fileMatch[2];
  
  // Extract clean error message (remove file path and line number)
  let errorMessage = firstErrorLine
    .replace(/\[ERROR\]/g, '')
    .replace(filePath, '')
    .replace(/\[\d+,\d+\]/g, '')
    .replace(/:\d+:/g, ':')
    .trim();

  // Get more context from subsequent lines if needed
  if (errorBlock.length > 1 && errorMessage.length < 50) {
    const secondLine = errorBlock[1].replace(/\[ERROR\]/g, '').trim();
    if (secondLine) {
      errorMessage += ' ' + secondLine;
    }
  }

  return {
    filePath,
    lineNumber: parseInt(lineNumber),
    message: errorMessage.trim(),
    rawError: errorBlock.join('\n')
  };
}

module.exports = { parseLogForError };