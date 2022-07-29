
function show_tab2() {
    console.log("element display");
    var getTab = document.getElementById("template1");
    getTab.classList.remove('active');
    var getTab3 = document.getElementById("template3");
    getTab3.classList.remove("active");
    var getTab2 = document.getElementById("template2");
    getTab2.classList.add("active");

    var get_body1 = document.getElementById("template1_body");
    var get_body2 = document.getElementById("template2_body");
    var get_body3 = document.getElementById("template3_body");

    get_body1.classList.remove('active');
    get_body3.classList.remove('active');
    get_body2.classList.add('active');

}

function show_tab1() {
    var getTab = document.getElementById("template2");
    getTab.classList.remove('active');
    var getTab3 = document.getElementById("template3");
    getTab3.classList.remove("active");
    var getTab2 = document.getElementById("template1");
    getTab2.classList.add("active");

    var get_body1 = document.getElementById("template2_body");
    var get_body2 = document.getElementById("template1_body");
    var get_body3 = document.getElementById("template3_body");

    get_body1.classList.remove('active');
    get_body3.classList.remove('active');
    get_body2.classList.add('active');

}

function show_tab3() {

    var getTab = document.getElementById("template2");
    getTab.classList.remove('active');
    var getTab1 = document.getElementById("template1");
    getTab1.classList.remove('active');
    var getTab3 = document.getElementById("template3");
    getTab3.classList.add("active");

    var get_body1 = document.getElementById("template2_body");
    var get_body2 = document.getElementById("template1_body");
    var get_body3 = document.getElementById("template3_body");

    get_body1.classList.remove('active');
    get_body2.classList.remove('active');
    get_body3.classList.add('active');

}

function submit_template() {

    $.ajax({
            url: '/dashboard/temp',
            data: {
              'username': 'testing val'
            },
            dataType: 'json',
            success: function (data) {
                alert("success!!");

            }
          });


}

function source_code_gen() {
    console.log('this works');
    var get_code = document.getElementById('t2_id').value;
    var get_code_body = document.getElementById('t2_function').value;

    console.log(get_code);

    $.ajax({
            url: '/dashboard/ssh_model',
            data: {
              'annotation_method': get_code,
              'annotation_body' : get_code_body
            },
            dataType: 'json',
            success: function (data) {
                console.log(data);
                alert(data['text']);

            }
          });
}


function source_code_gen3() {
    console.log('this works');
    var get_code = document.getElementById('t3_id').value;
    var get_code_body = document.getElementById('t3_function').value;

    console.log(get_code);

    $.ajax({
            url: '/dashboard/ssh_model',
            data: {
              'annotation_method': get_code,
              'annotation_body' : get_code_body
            },
            dataType: 'json',
            success: function (data) {
                console.log(data);
                alert(data['text']);
//                window.location.href = "/dashboard/display_results";


            }
          });
}


//function display(data) {
//
//    $.ajax({
//            url: '/dashboard/display_results',
//            data: {
//              'annotation_method': get_code,
//
//            },
//            dataType: 'json',
//            success: function (data) {
//                console.log(data);
//                alert(data['text']);
//
//            }
//          });
//
//}