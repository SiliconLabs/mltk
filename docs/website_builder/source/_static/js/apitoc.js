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

    var props, methods, funcs, variables;
    headings.each((index) => {
        var element = headings[index];
        if(element.innerText == 'Properties') {
            props = $(element.nextElementSibling).find('a');
        } else if(element.innerText == 'Methods') {
            methods = $(element.nextElementSibling).find('a');
        } else if(element.innerText == 'Functions') {
            funcs = $(element.nextElementSibling).find('a');
        } else if(element.innerText == 'Variables') {
            variables = $(element.nextElementSibling).find('a');
        }
    });

    function getHref(el) {
        return $(el).attr('href').trim();
    }

    function getTitle(el) {
        var toks = el.title.split('.');
        return toks[toks.length-1].trim();
    }

    if(props && !window._added_props) {
        window._added_props = true;
        var s = '<li class="md-nav__item"><label class="md-nav__title" style="padding-top:30px">Properties</label>' +
        '<nav class="md-nav"><ul class="md-nav__list">';
        props.each(index => {
            var element = props[index];
            s += `<li class="md-nav__item"><a href="${getHref(element)}" class="md-nav__link">${getTitle(element)}</a></li>`
        })
        s += '</ul></nav></li>';
        localtoc.append($($.parseHTML(s)));
    }

    if(methods && !window._added_methods) {
        window._added_methods = true;
        var s = '<li class="md-nav__item"><label class="md-nav__title" style="padding-top:30px">Methods</label>' +
        '<nav class="md-nav"><ul class="md-nav__list">';
        methods.each(index => {
            var element = methods[index];
            s += `<li class="md-nav__item"><a href="${getHref(element)}" class="md-nav__link">${getTitle(element)}</a></li>`
        })
        s += '</ul></nav></li>';
        localtoc.append($($.parseHTML(s)))
    }

    if(variables && !window._added_variables) {
        window._added_variables = true;
        var s = '<li class="md-nav__item"><label class="md-nav__title" style="padding-top:30px">Variables</label>' +
        '<nav class="md-nav"><ul class="md-nav__list">';
        variables.each(index => {
            var element = variables[index];
            s += `<li class="md-nav__item"><a href="${getHref(element)}" class="md-nav__link">${getTitle(element)}</a></li>`
        })
        s += '</ul></nav></li>';
        localtoc.append($($.parseHTML(s)))
    }

    if(funcs && !window._added_funcs) {
        window._added_funcs = true;
        var s = '<li class="md-nav__item"><label class="md-nav__title" style="padding-top:30px">Functions</label>' +
        '<nav class="md-nav"><ul class="md-nav__list">';
        funcs.each(index => {
            var element = funcs[index];
            s += `<li class="md-nav__item"><a href="${getHref(element)}" class="md-nav__link">${getTitle(element)}</a></li>`
        })
        s += '</ul></nav></li>';
        localtoc.append($($.parseHTML(s)))
    }

    if(!window._remove_toc && (window._added_props || window._added_methods || window._added_variables || window._added_funcs)) {
        window._remove_toc = true;
        localtoc.children().first().remove();
    }

});