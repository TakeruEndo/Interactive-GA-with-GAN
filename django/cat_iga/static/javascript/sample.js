window.onload = function() {
    var changeColor = function() {
        var e = document.getElementById('test');
        e.style.color = 'red';
        console.log("書き換えテスト")
    }
    setTimeout(changeColor, 5000);
}


function clk(el){
    /// 要素IDを取得する
    var e = el || window.event;
    var elemId = el.id;
    var count = el.detail;
    console.log(elemId, count);
    // var el_style = el.style.border     
    el.style.border = "3px solid blue";
    // switch(buttonId){
    //     case 'b1': 
    //         break;
    //     case 'b2': 
    //         break;
    //     case 'b3':
    //         break;
    // }
}