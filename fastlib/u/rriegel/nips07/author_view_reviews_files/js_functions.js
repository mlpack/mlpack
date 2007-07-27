// $_CVS; $_CVS .= '$Id: js_functions.js,v 1.0 2004-02-05 09:39:13+01 preuss Exp preuss $' . "\n";

// ------------------------------------------------------------------
// General Settings
FormIsChanged=false
// ------------------------------------------------------------------

// ------------------------------------------------------------------
function CheckAllRadio(FormName, Wert) {
	var AnzRadio = FormName.length;
	for (var i=0; i<AnzRadio; i++) {
    if ((FormName.elements[i].type=="radio") && (parseInt(FormName.elements[i].value)==Wert)) {
      FormName.elements[i].checked = true;
      RadioButtonChanged(FormName.elements[i]);
    };
	}
}
// ------------------------------------------------------------------
function ResetRadioForm(FormName) {
	FormName.reset();
	var AnzRadio = FormName.length;

	for (var i=0; i<AnzRadio; i++) {
    if (FormName.elements[i].checked) {
      IdName = "Previous_".concat(FormName.elements[i].name).replace(/Paper\[/,'Interest_').replace(/\]/,'');
      document.getElementById(IdName).value = FormName.elements[i].value;
    };
	};
  return false;
}
// ------------------------------------------------------------------
function RadioButtonChanged(element){
    ev        = parseInt(element.value);
    IdName    = "Previous_".concat(element.name).replace(/Paper\[/,'Interest_').replace(/\]/,'');
    oldValue  = parseInt(document.getElementById(IdName).value);

    switch (oldValue) {
      case  2: document.Interest.CtrVeryWish.value-- ; break;
      case  1: document.Interest.CtrWish.value--     ; break;
      case -1: document.Interest.CtrDislike.value--  ; break;
      case -2: document.Interest.CtrConflict.value-- ; break;
    };
  
    switch (ev) {
      case  2: document.Interest.CtrVeryWish.value++ ; break;
      case  1: document.Interest.CtrWish.value++     ; break;
      case -1: document.Interest.CtrDislike.value++  ; break;
      case -2: document.Interest.CtrConflict.value++ ; break;
    };

    document.getElementById(IdName).value = ev;
  	return true;
};

// ------------------------------------------------------------------
function OpenNewWindow(URL) {
	w1 = open(	URL, 'SecondWindow','width=840,height=450,left=150,top=200,resizable=yes,dependent=yes,location=no,menubar=no,scrollbars=yes,status=no,toolbar=no');
	w1.focus();
	return false;
};

// ------------------------------------------------------------------
function CfC() { // check for changes
		if (FormIsChanged)
			return (window.confirm("You did not save your changes. Do you want to continue without saving?"));
		return true;
};
// ------------------------------------------------------------------
function OpenNewPrintWindow(URL) {
	w2 = open(	URL, 'PrintWindow','width=770,height=450,left=100,top=150,resizable=yes,dependent=yes,location=no,menubar=yes,scrollbars=yes,status=no,toolbar=yes');
	w2.focus();
	return false;
};
// ------------------------------------------------------------------
function OpenNewHelpWindow(URL) {
	w2 = open(	URL, 'HelpWindow','width=770,height=450,left=100,top=150,resizable=yes,dependent=yes,location=no,menubar=no,scrollbars=yes,status=no,toolbar=no');
	w2.focus();
	return false;
};
// ------------------------------------------------------------------

function FormatWindow() {
	// window.innerHeight = 300;
	// window.innerWidth  = 600;
	window.resizeTo(800,600);
	window.focus();
};

// ------------------------------------------------------------------

function ClearAll(FormName) {
	var AnzCheckboxes = FormName.length;
	for (var i=0; i<AnzCheckboxes; i++) {
		if (FormName.elements[i].type=="checkbox") FormName.elements[i].checked = false;
	}
}

// ------------------------------------------------------------------

function ClearAllTextFields(FormName) {
	var AnzFields = FormName.length;
	for (var i=0; i<AnzFields; i++) {
		if (FormName.elements[i].type=="text") FormName.elements[i].value = '';
	}
}

// ------------------------------------------------------------------

function CheckAll(FormName) {
	var AnzCheckboxes = FormName.length;
	for (var i=0; i<AnzCheckboxes; i++) {
		if (FormName.elements[i].type=="checkbox") FormName.elements[i].checked = true;
	}
}

// ------------------------------------------------------------------

function SetUrlAndSubmit(ZielURL) {
    document.forms['DynamicForm'].action = ZielURL;
    document.forms['DynamicForm'].submit();
    return false; 
}



