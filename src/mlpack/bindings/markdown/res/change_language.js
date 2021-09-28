/**
 * @file change_language.js
 * @author Ryan Curtin
 *
 * A utility function to change the language displayed on the page.  This
 * function should be called whenever the language is changed from the
 * drop-down.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
function changeLanguage()
{
  lang = document.getElementById("language-select").value;
  var links = document.getElementsByClassName("language-link");
  for (i = 0; i < links.length; ++i)
  {
    // With each of the links, we get the inner <a>, but we need the parent
    // <li>.
    if (links[i].id == lang || links[i].id == "always")
      links[i].parentElement.style.display = "list-item";
    else
      links[i].parentElement.style.display = "none";
  }

  var titles = document.getElementsByClassName("language-title");
  for (i = 0; i < titles.length; ++i)
  {
    if (titles[i].id == lang)
      titles[i].style.display = "inline";
    else
      titles[i].style.display = "none";
  }

  var headers = document.getElementsByClassName("language-header");
  for (i = 0; i < headers.length; ++i)
  {
    if (headers[i].id == lang)
      headers[i].style.display = "inline";
    else
      headers[i].style.display = "none";
  }

  var decls = document.getElementsByClassName("language-decl");
  for (i = 0; i < decls.length; ++i)
  {
    if (decls[i].id == lang)
      decls[i].style.display = "inline";
    else
      decls[i].style.display = "none";
  }

  var types = document.getElementsByClassName("language-types");
  for (i = 0; i < types.length; ++i)
  {
    if (types[i].id == lang)
      types[i].style.display = "inline";
    else
      types[i].style.display = "none";
  }

  var details = document.getElementsByClassName("language-detail-link");
  for (i = 0; i < details.length; ++i)
  {
    if (details[i].id == lang)
      details[i].style.display = "inline";
    else
      details[i].style.display = "none";
  }

  var sections = document.getElementsByClassName("language-section");
  for (i = 0; i < sections.length; ++i)
  {
    if (sections[i].id == lang)
      sections[i].style.display = "inline";
    else
      sections[i].style.display = "none";
  }
}

document.body.onload = function()
{
  // Do we need to manually set the language because the user came with an
  // anchor?
  if (window.location.hash)
  {
    // Try to extract the language.
    firstUnderscore = window.location.hash.indexOf("_");
    if (firstUnderscore !== -1)
    {
      var lang = window.location.hash.substring(1, firstUnderscore);
      // Now see if it's in the list of languages.
      var select = document.getElementById("language-select");
      for (i = 0; i < select.length; ++i)
      {
        var select_lang = select[i].value;
        // Is the language a match?
        if (lang === select_lang)
        {
          select.value = select_lang;
          break;
        }
      }
    }
  }

  changeLanguage();
}
