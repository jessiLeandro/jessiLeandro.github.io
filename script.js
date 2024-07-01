function includeHTML() {
    var element = document.getElementById("common-content");
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "common.html", true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            element.innerHTML = xhr.responseText;
        }
    };
    xhr.send();
}
