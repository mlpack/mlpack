// wrap_lines.js: find method signatures in mlpack documentation that are too
// long to be displayed, and wrap them in a reasonably smart way.

/**
 * Remove any line wrapping that has already been done.
 */
function clearWraps()
{
  const wraps = [...document.getElementsByClassName("signature_wrap")];
  wraps.forEach((e) => e.remove());
}

/**
 * Insert a <br> and <span> with spaces to wrap the <code> element `signature`
 * at its child index `insertIndex`.  Use `numSpaces` as the number of spaces
 * for the next indented line.
 */
function insertWrap(signature, insertIndex, numSpaces)
{
  let spaces = document.createElement("span");
  spaces.classList.add("signature_wrap");
  spaces.textContent = " ".repeat(numSpaces);
  signature.insertBefore(spaces, signature.childNodes[insertIndex]);

  let br = document.createElement("br");
  br.classList.add("signature_wrap");
  signature.insertBefore(br, spaces);
}

/**
 * Wrap the line `signature`, assuming that it is too wide and needs to be
 * wrapped.  `signature` is expected to be the <code> element containing the
 * entire signature, and `maxNumChars` is the maximum number of characters we
 * allow on any single line.
 *
 * We try to wrap like this:
 *
 * x = method(a, b, c=100, d=500)
 *
 * x = method(a, b,
 *            c=100,
 *            d=100)
 *
 * i.e. align wrapped arguments with the opening parenthetical.
 *
 * If the average number of parameters per line is less than 2.5, and
 * there is more than one line, then put one parameter per line.
 *
 * If the text is so narrow that we can't even keep the first parameter on
 * its own line, then wrap as
 *
 * x = method(
 *       a,
 *       b,
 *       c=100,
 *       d=100)
 *
 * And if even that doesn't fit, well, the text must be so narrow that
 * we'll just give up and let it be over-wide.
 */
function wrapLine(signature, maxNumChars)
{
  let firstParen = signature.textContent.indexOf('(');
  if (!firstParen)
  {
    console.log("Could not find opening parenthesis of signature: '" +
        signature.textContent + "'");
    return;
  }

  const commas = [...signature.textContent.matchAll(/, /g)];
  if (commas.length == 0)
    return;

  // We need to compute the locations of all of the first characters *after*
  // a comma and a space.
  let argLocs = []
  let associatedChildNodeIndex = 0;
  let totalNumChars = 0;
  while ((totalNumChars < firstParen + 1) &&
         (associatedChildNodeIndex < signature.childNodes.length))
  {
    totalNumChars += signature.childNodes[
        associatedChildNodeIndex].textContent.length;
    ++associatedChildNodeIndex;
  }

  // Paranoia, although this should never happen.
  if (associatedChildNodeIndex == signature.childNodes.length)
  {
    console.log("Could not find child node index for opening parenthesis of " +
        "signature: '" + signature.textContent + "'!");
    return;
  }

  argLocs.push([firstParen + 1, commas[0].index + 1,
      associatedChildNodeIndex]);

  for (let j = 0; j < commas.length; ++j)
  {
    while ((totalNumChars < commas[j].index + 2) &&
           (associatedChildNodeIndex < signature.childNodes.length))
    {
      totalNumChars += signature.childNodes[
          associatedChildNodeIndex].textContent.length;
      ++associatedChildNodeIndex;
    }

    if (commas[j].index > firstParen &&
        commas[j].index + 2 < signature.textContent.length)
    {
      const argStart = commas[j].index + 2;
      const argEnd = (j == commas.length - 1) ?
          signature.textContent.length : (commas[j + 1].index + 1);

      argLocs.push([argStart, argEnd, associatedChildNodeIndex]);
    }
  }

  // First check whether we need to wrap the very first argument---if so,
  // then we have to wrap in a different way.
  let mustWrap = false;
  if (argLocs[0][1] >= maxNumChars)
  {
    mustWrap = true;
    firstParen = 3; // so that we get 4-space indentation when we wrap.
  }

  // Keep track of which line each argument is on.
  let argLines = []
  argLines.length = argLocs.length;
  argLines.fill(0);

  // First try a really simple strategy...
  for (let j = 0; j < argLocs.length; ++j)
  {
    const argStart = argLocs[j][0];
    const argEnd = argLocs[j][1];
    const argNodeIndex = argLocs[j][2];

    if (argEnd >= maxNumChars || mustWrap)
    {
      insertWrap(signature, argNodeIndex, firstParen + 1);

      argLocs[j][2] += 2;
      argLines[j] += 1;

      // Since this argument is wrapped, we now need to update the locations
      // of the rest of the array.
      const charAdjustment = (firstParen + 1) - argStart;
      for (let k = j + 1; k < argLocs.length; ++k)
      {
        argLocs[k][0] += charAdjustment;
        argLocs[k][1] += charAdjustment;
        argLocs[k][2] += 2;
        argLines[k] += 1;
      }
    }
  }

  // Now check the average number of arguments per line.  If it's less than
  // three, then we need to wrap everything else too.
  let argsPerLine = argLines.length / (argLines[argLines.length - 1] + 1);
  if (argsPerLine < 2.5)
  {
    let insertedNodes = 0;
    for (let j = 1; j < argLocs.length; ++j)
    {
      // If the argument hasn't been wrapped yet, force-wrap it.
      if (argLines[j] == argLines[j - 1])
      {
        insertWrap(signature, argLocs[j][2] + insertedNodes,
            firstParen + 1);
        insertedNodes += 2;
      }
    }
  }
}

function wrapLines()
{
  // First, clear all line wrapping we may have done on a previous call.
  clearWraps();

  // Find all <code> blocks inside of an <li> inside the content div.
  const candidates = document.querySelectorAll("div#content > ul > li > code");
  if (candidates.length == 0)
    return;

  // Not all <code> blocks inside of an <li> are what we're looking for.
  // Specifically we want to find <code> blocks that are the *full* text of an
  // <li> that is a child of a top-level <ul>.
  const candidatesArray = Array.from(candidates);
  const signatures = candidatesArray.filter((c) => {
      const p = c.parentNode;

      // The <li> must have no other children, or, if it does, it needs to be a
      // child <ul>.
      if (p.children.length > 2)
        return false;
      if (p.children[0] != c)
        return false;
      if (p.children.length == 2 &&
          p.children[1].nodeName.toLowerCase() != "ul")
        return false;

      // There must not be any extra text in the li that's not a part of either
      // of its elements.  This turns out to be a bit tedious to check: we need
      // to look for any text-only .childNodes that aren't whitespace.
      let childIndex = 0;
      for (let i = 0; i < p.childNodes.length; ++i)
      {
        if (p.childNodes[i].nodeName.toLowerCase() == "#text")
        {
          if (p.childNodes[i].data.trim().length != 0)
          {
            // Non-whitespace in a text node.
            return false;
          }
        }
        else if (p.childNodes[i] === p.children[childIndex])
        {
          ++childIndex;
        }
        else
        {
          // Got an unexpected non-#text non-child node.
          console.log(i + "Unexpected non-#text non-child node!");
          return false;
        }
      }

      return true;
  });

  // Now we have a bunch of things that look to be signatures that we could
  // wrap.  Iterate through them, and find those that are too long.
  const contentDiv = document.querySelector("div#content");
  if (!contentDiv)
  {
    console.log("Failed to find div#content for method signature wrapping!");
    return; // Failure to find the content div...
  }
  const contentDivLeftPadding =
      getComputedStyle(contentDiv).paddingLeft.replace(/px$/, '');
  const contentDivRightPadding =
      getComputedStyle(contentDiv).paddingRight.replace(/px$/, '');;
  if (isNaN(parseInt(contentDivLeftPadding)) ||
      isNaN(parseInt(contentDivRightPadding)))
  {
    console.log("Could not compute left or right padding of div#content!");
    return;
  }

  // This is the overall width of the #content div.
  const contentWidth = contentDiv.clientWidth -
      parseInt(contentDivLeftPadding) -
      parseInt(contentDivRightPadding);

  for (let i = 0; i < signatures.length; ++i)
  {
    // Compute the total width that the signature is allowed to have; this is
    // the overall width of the #content div with any padding from the parent ul
    // and li removed (that includes the bullet point marker).
    const parentLiWidth = signatures[i].parentNode.clientWidth;
    const parentUlWidth = signatures[i].parentNode.parentNode.clientWidth;
    const ulMarkerPixels = parentUlWidth - parentLiWidth;
    const usableWidth = contentWidth - ulMarkerPixels;

    if (signatures[i].scrollWidth > usableWidth)
    {
      // This signature is overflowing.

      // To compute the width of each character, we must remove the padding from
      // the <code> block (if any).
      const signatureStyle = getComputedStyle(signatures[i]);
      const signatureLeftPadding =
          parseInt(signatureStyle.paddingLeft.replace(/px$/, '')) || 0;
      const signatureRightPadding =
          parseInt(signatureStyle.paddingRight.replace(/px$/, '')) || 0;

      // This is the total width of the characters that are being displayed.
      const effectiveWidth = signatures[i].scrollWidth - signatureLeftPadding -
          signatureRightPadding;

      // And now this is the width of each individual character.
      const charWidth = effectiveWidth / signatures[i].textContent.length;

      // Get the number of characters we could fit without overflowing.
      const maxNumChars = Math.floor(usableWidth / charWidth);

      wrapLine(signatures[i], maxNumChars);
    }
  }
}

window.addEventListener("load", (event) => { wrapLines(); });
window.addEventListener("resize", (event) => { wrapLines(); });
