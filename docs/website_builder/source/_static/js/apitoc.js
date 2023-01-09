// Inject API members into the TOC in the sidebar.
// This should be loaded in the localtoc.html template.

$(function (){
    var localtoc = $('#localtoc');
    if(!localtoc) {
        return;
    }

    var headings = $('p[class="rubric"]');
    if(!headings) {
        return;
    }

    var props, methods, funcs;
    headings.each((index) => {
        var element = headings[index];
        if(element.innerText == 'Properties') {
            props = $(element.nextElementSibling).find('a');
        } else if(element.innerText == 'Methods') {
            methods = $(element.nextElementSibling).find('a');
        } else if(element.innerText == 'Functions') {
            funcs = $(element.nextElementSibling).find('a');
        }
    });

    function getHref(el) {
        return $(el).attr('href').trim();
    }

    function getTitle(el) {
        var toks = el.title.split('.');
        return toks[toks.length-1].trim();
    }

    if(props) {
        var s = '<li class="md-nav__item"><label class="md-nav__title" style="padding-top:30px">Properties</label>' +
        '<nav class="md-nav"><ul class="md-nav__list">';
        props.each(index => {
            var element = props[index];
            s += `<li class="md-nav__item"><a href="${getHref(element)}" class="md-nav__link">${getTitle(element)}</a></li>`
        })
        s += '</ul></nav></li>';
        localtoc.append($($.parseHTML(s)));
    }

    if(methods) {
        var s = '<li class="md-nav__item"><label class="md-nav__title" style="padding-top:30px">Methods</label>' +
        '<nav class="md-nav"><ul class="md-nav__list">';
        methods.each(index => {
            var element = methods[index];
            s += `<li class="md-nav__item"><a href="${getHref(element)}" class="md-nav__link">${getTitle(element)}</a></li>`
        })
        s += '</ul></nav></li>';
        localtoc.append($($.parseHTML(s)))
    }

    if(funcs) {
        var s = '<li class="md-nav__item"><label class="md-nav__title" style="padding-top:30px">Functions</label>' +
        '<nav class="md-nav"><ul class="md-nav__list">';
        funcs.each(index => {
            var element = funcs[index];
            s += `<li class="md-nav__item"><a href="${getHref(element)}" class="md-nav__link">${getTitle(element)}</a></li>`
        })
        s += '</ul></nav></li>';
        localtoc.append($($.parseHTML(s)))
    }


});