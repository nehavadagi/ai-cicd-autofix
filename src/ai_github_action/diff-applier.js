/**
 * Applies unified diff patches to source code
 */

const diff = require('diff');

function applyDiff(originalContent, diffText) {
  try {
    // Parse the unified diff
    const patches = diff.parsePatch(diffText);
    
    if (patches.length === 0) {
      throw new Error('No valid diff patches found');
    }

    // Apply the first patch (assuming single file diff)
    const result = diff.applyPatch(originalContent, patches[0]);
    
    if (result === false) {
      throw new Error('Failed to apply diff patch');
    }

    return result;
  } catch (error) {
    throw new Error(`Diff application failed: ${error.message}`);
  }
}

module.exports = { applyDiff };