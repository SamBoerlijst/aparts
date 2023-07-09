---

tags: Journal

cssclass: null

---

  
  
  

```dataviewjs

let name = dv.current().file.name

let postg = ")";

let preg = "[Google](https://www.google.com/search?q="

  

let namepercent = name.split(" ").join("%20%");

let nameplus = name.split(" ").join("+");

  

let google = preg + nameplus + postg;

  

let links = google

  

dv.header(2, name);

dv.span(links)

```

____

  

####  Indexed authors by frequency

```dataviewjs
let currentjournal = dv.current().file.name;
let articles = dv.pages("#article");
let output = "";
let state = "";
let journalstate = "";
let tempname = "";
let tempstate = "";
let authorlist = "";
let statelist;
let i = 0;
let j = 0;
let counts = [];
let authorlistfiltered;

for(let group of articles.groupBy(p => p.file.name)){
	tempname = articles[i].file.name;
	state = articles[i].author;
	journalstate = articles[i].journal;
	journalstate = journalstate + ", " + ""
	journalstate = journalstate.split(", ").join();
	state = state + " " + ""
	state = state.split(", ");
	state = state.join("");
	tempstate = journalstate.includes(currentjournal);
	if(tempstate == 1){
		authorlist = authorlist + ", " + state;
	}else{authorlist = authorlist};
	statelist = statelist + ", " + tempstate;
	i = i+1;
	};
	
authorlist = authorlist.replace("    ", " ").replace("   ", " ").replace("  ", " ");
authorlist = authorlist.split(",");
authorlist = authorlist.sort()
authorlist = authorlist.slice(1);
authorlistfiltered = authorlist.filter((v, j, a) => a.indexOf(v) === j);

counts = authorlist.reduce((acc, curr) => (acc[curr] = (acc[curr] || 0) + 1, acc), {});
counts = Object.values(counts)



//combine and sort arrays by count
var object = []
for (j = 0; j < authorlistfiltered.length; j++) {
    object.push({'name': authorlistfiltered[j], 'amount': counts[j]});}

object.sort(function(a, b) {
    return ((a.amount > b.amount) ? -1 : ((a.amount == b.amount) ? 0 : 1));
});

//output
for (i = 0; i<object.length; i++) {
    let out = object[i].name + "&nbsp" + object[i].amount + ",&nbsp";
	output = output + out
}
output = output.substring(0, output.length-6)
dv.span(output)

```

____

  

#### Common Tags

```dataviewjs
let currentjournal = dv.current().file.name;
let articles = dv.pages("#article");

let counts;
let fullist;
let fullistfiltered;
let state;
let statelist;
let tags = "";
let taglist;
let tempstate;
let i = 0;
let j = 0;

//select articles that contain the relevant journal and retrieve tags

for(let group of articles.groupBy(p => p.file.name)){
    taglist = articles[i].tags;
    state = articles[i].journal;
    state = state + ", " + ""
    state = state.split(", ");
    state = state.join();
    tempstate = state.includes(currentjournal);
    if(tempstate == 1){
        fullist = fullist + " " + taglist;
    }else{fullist = fullist};
    statelist = statelist + " " + tempstate;
    i = i+1;
};

  

// filter uniques

fullist = fullist.replace(",", "");
fullist = fullist.split(" ");
fullist = fullist.slice(1);
fullist = fullist.sort();
fullistfiltered = fullist.filter((v, j, a) => a.indexOf(v) === j);

counts = fullist.reduce((acc, curr) => (acc[curr] = (acc[curr] || 0) + 1, acc), {});
counts = Object.values(counts);

//convert to tags

for(j=0; j<fullistfiltered.length; j++){
    fullistfiltered[j] = "#" + fullistfiltered[j]
}

  

//combine and sort arrays by count

var object = []

for (j = 0; j < fullistfiltered.length; j++) {
    object.push({'name': fullistfiltered[j], 'amount': counts[j]});}

object.sort(function(a, b) {
    return ((a.amount > b.amount) ? -1 : ((a.amount == b.amount) ? 0 : 1));
});

//output

for (i = 0; i<object.length; i++) {
    let out = object[i].name + " ";
    dv.span(out)

}

```

____

  
  

#### Associated Papers

  

```dataviewjs
let currentauthor = dv.current().file.name;
let articles = dv.pages("#article");

let authorlist;
let card = "";
let cmlist;
let filelist;
let i = 0;
let j = 0;
let journallist;
let state;
let statelist;
let titlelist;
let tempcm;
let tempfile;
let tempjournal;
let tempname;
let tempstate;
let tempyear;
let yearlist;

for(let group of articles.groupBy(p => p.file.name)){
	tempname = articles[i].file.name;
	tempcm = articles[i].CoreMessage;
	tempfile = articles[i].files;
	tempjournal = articles[i].journal;
	tempyear = articles[i].year;
	
	state = articles[i].journal;
	state = state + ", " + ""
	state = state.split(", ").join();
	tempstate = state.includes(currentauthor);
	
	if(tempstate == 1){
		authorlist = authorlist + ", " + state;
		titlelist = titlelist + ", " + tempname;
		cmlist = cmlist + ", " + tempcm + "\n";
		filelist = filelist + ", " + tempfile;
		journallist = journallist + ", " + tempjournal;
		yearlist = yearlist + ", " + tempyear;
	};
	statelist = statelist + ", " + tempstate;
	i = i+1;
};

authorlist = authorlist.split(", ");
titlelist = titlelist.split(",");
cmlist = cmlist.split(",");
filelist = filelist.split(",");
journallist = journallist.split(",");
yearlist = yearlist.split(",");

for(i = 1; i<titlelist.length; i++){
titlelist[i] = "[[" + titlelist[i] + "]]"
}

for(j = 1; j<titlelist.length; j++){
	card = cmlist[j] + authorlist[j] + "\nYear:&nbsp;" + yearlist[j] + "&nbsp;&nbsp;&nbsp;&nbsp;Journal:&nbsp;" + journallist[j] + "&nbsp;&nbsp;&nbsp;&nbsp;PDF:&nbsp;" + filelist[j] + "\n\n";
	dv.header(6,titlelist[j]);
	dv.span(card)
};
```
