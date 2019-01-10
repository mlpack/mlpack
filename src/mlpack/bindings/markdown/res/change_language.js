/**
 * A utility function to change the language displayed on the page.  This
 * function should be called whenever the language is changed from the
 * drop-down.
 */
function changeLanguage()
{
  lang = document.getElementById("language-select").value;
  var links = document.getElementsByClassName("language-link");
  for (i = 0; i < links.length; ++i)
  {
    // With each of the links, we get the inner <a>, but we need the parent
    // <li>.
    if (links[i].id == lang)
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

  var sections = document.getElementsByClassName("language-section");
  for (i = 0; i < sections.length; ++i)
  {
    if (sections[i].id == lang)
      sections[i].style.display = "inline";
    else
      sections[i].style.display = "none";
  }
}

document.body.onload = changeLanguage()
