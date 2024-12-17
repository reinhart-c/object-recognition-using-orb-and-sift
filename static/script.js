function modelSelect(elem) {
    console.log("/get_result/" + elem.id);
    document.getElementById("resultImage").src = "/get_result/" + elem.id;
}

document.getElementById("resultImage").src = "/get_result/ORB";
